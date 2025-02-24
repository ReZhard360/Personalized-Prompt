import os
import requests
from bs4 import BeautifulSoup
from bs4.element import Comment
from urllib.parse import urlparse, urljoin
import csv
import time
import json
import tiktoken
import folium
import gradio as gr
import logging

# Configurazione del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import per nltk (stopwords italiane)
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
italian_stopwords = stopwords.words('italian')

# Import Elasticsearch e configurazione tramite variabili d'ambiente
from elasticsearch import Elasticsearch

ES_URL = os.getenv("ES_URL", "http://localhost:9200")
ES_USER = os.getenv("ES_USER", "elastic")
ES_PASSWORD = os.getenv("ES_PASSWORD", "iD8R41p-GsP5UW*y4AIb")
VERIFY_CERTS = os.getenv("VERIFY_CERTS", "False").lower() == "true"
INDEX_NAME = os.getenv("INDEX_NAME", "langchain_index")

# Usa basic_auth per evitare warning di deprecazione
es = Elasticsearch(
    ES_URL,
    basic_auth=(ES_USER, ES_PASSWORD),
    verify_certs=VERIFY_CERTS
)

# Import del modello LLM tramite Ollama e dello splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM

# Import metriche (BERTScore, ROUGE)
from bert_score import score as bert_score
from rouge_score import rouge_scorer

########################################
# CONFIGURAZIONE GLOBALI E UTILITY
########################################

language_instructions = {
    "English": "Write the summary in English.",
    "Chinese": "用中文撰写摘要。",
    "Hindi": "हिंदी में सारांश लिखें।",
    "Spanish": "Escribe il riassunto in español.",
    "Arabic": "اكتب الملخص باللغة العربية.",
    "Italian": "Scrivi il riassunto in italiano."
}

narrator_instructions = {
    "Tour Guide": "Immagina di essere una guida turistica esperta, capace di raccontare con passione e semplicità.",
    "Historian": "Adotta uno stile da storico appassionato, che contestualizza i fatti nel tempo.",
    "Friendly": "Usa un tono amichevole e accessibile, come se parlassi con un amico.",
    "Expert": "Sii preciso ed esperto, con un tono formale e autorevole.",
    "Storyteller": "Racconta la storia in modo coinvolgente, come un narratore che incanta il pubblico."
}

# Impostazioni LLM (default "llama3.2")
DEFAULT_MODEL = "llama3.2"
llm = OllamaLLM(
    base_url="http://localhost:11434",
    model=DEFAULT_MODEL,
    temperature=0.7
)

def update_llm_model(model_name: str):
    logging.info(f"Aggiorno il modello LLM a: {model_name}")
    llm.model = model_name

def count_tokens(text: str) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

########################################
# FUNZIONI DI SUPPORTO PER LO SCRAPING (COMUNI)
########################################

def is_valid_url(url: str) -> bool:
    parsed = urlparse(url)
    return bool(parsed.scheme and parsed.netloc)

def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

def extract_text_from_url(url: str) -> str:
    if not is_valid_url(url):
        return "Invalid URL"
    try:
        headers = {'User-Agent': 'Your Webscraper Bot 1.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        texts = soup.find_all(string=True)
        visible_texts = filter(tag_visible, texts)
        return " ".join(t.strip() for t in visible_texts)
    except Exception as e:
        return f"Error: {str(e)}"

def parse_location(location_str: str):
    parts = [p.strip() for p in location_str.split(',')]
    city = parts[0] if len(parts) > 0 else ""
    province = parts[1] if len(parts) > 1 else ""
    region = parts[2] if len(parts) > 2 else ""
    return region, province, city

########################################
# FUNZIONI DI SCRAPING DA FONTI TRADIZIONALI (WIKIPEDIA, CULTURA.GOV.IT)
########################################

def extract_monument_data(url: str, monument_type: str) -> dict:
    data = {
        "name": "",
        "description": "",
        "region": "",
        "province": "",
        "city": "",
        "lat": "",
        "lon": "",
        "type": monument_type,
        "source": "",
        "url": url
    }
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        title_tag = soup.find("h1", id="firstHeading")
        if title_tag:
            data["name"] = title_tag.text.strip()
        content_div = soup.find("div", id="mw-content-text")
        if content_div:
            paragraphs = content_div.find_all("p")
            desc = ""
            for para in paragraphs:
                text = para.get_text(strip=True)
                if text:
                    desc += text + " "
                if len(desc) > 300:
                    break
            data["description"] = desc.strip()
        infobox = soup.find("table", class_="infobox")
        loc_text = ""
        if infobox:
            for row in infobox.find_all("tr"):
                header = row.find("th")
                if header and ("localizzazione" in header.text.lower() or "ubicazione" in header.text.lower()):
                    td = row.find("td")
                    if td:
                        loc_text = td.get_text(separator=",", strip=True)
                        break
        if loc_text:
            data["region"], data["province"], data["city"] = parse_location(loc_text)
        coord_tag = soup.find("span", class_="geo-dec")
        if not coord_tag:
            coord_tag = soup.find("span", class_="geo")
        if coord_tag:
            coords = coord_tag.text.strip().split()
            if len(coords) >= 2:
                try:
                    lat = coords[0].replace("°N", "").replace("°S", "")
                    lon = coords[1].replace("°E", "").replace("°W", "")
                    data["lat"] = float(lat)
                    data["lon"] = float(lon)
                except Exception as ex:
                    data["lat"] = ""
                    data["lon"] = ""
        data["source"] = "Wikipedia" if "wikipedia.org" in url else urlparse(url).netloc
    except Exception as e:
        logging.error(f"Errore nell'estrazione dei dati da {url}: {e}")
    return data

def get_monuments_from_wikipedia(page_url: str, monument_type: str) -> list:
    monuments = []
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(page_url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            if href and href.startswith('/wiki/') and ':' not in href:
                full_url = urljoin("https://it.wikipedia.org", href)
                m_data = extract_monument_data(full_url, monument_type)
                if m_data.get("name"):
                    monuments.append(m_data)
                    time.sleep(0.5)
    except Exception as e:
        logging.error(f"Errore nello scraping della pagina {page_url}: {e}")
    return monuments

def get_all_monuments() -> list:
    cathedrals_url = "https://it.wikipedia.org/wiki/Categoria:Cattedrali_della_Sicilia"
    monuments = get_monuments_from_wikipedia(cathedrals_url, "Cattedrale")
    return monuments

def get_monuments_from_cultura() -> list:
    """
    Scraping dal sito Cultura.gov.it per "chiesa o edificio di culto" in Sicilia.
    """
    url = "https://cultura.gov.it/luoghi/cerca-luogo?regione=regione-sicilia&tipo=chiesa-o-edificio-di-culto"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        cards = soup.find_all("div", class_="luogo-card")
        results = []
        for card in cards:
            name_tag = card.find("h2") or card.find("h3")
            name = name_tag.text.strip() if name_tag else "N/D"
            desc_tag = card.find("p")
            description = desc_tag.text.strip() if desc_tag else ""
            data = {
                "name": name,
                "description": description,
                "region": "Sicilia",
                "province": "",
                "city": "",
                "lat": "",
                "lon": "",
                "type": "Chiesa o edificio di culto",
                "source": "Cultura.gov.it",
                "url": url
            }
            results.append(data)
        return results
    except Exception as e:
        logging.error(f"Errore nello scraping di Cultura.gov.it: {e}")
        return []

########################################
# FUNZIONI DI SCRAPING PER ALTRI SITI
########################################

def get_monuments_from_clicksicilia() -> list:
    """
    Scraping dal sito ClickSicilia per santuari ed eremi celebri.
    Ho aggiornato i selettori in base alla struttura osservata:
    - I blocchi degli elementi sono in tag <div> con classe "santuario-list-item".
    - Il titolo si trova in un tag <h3> e la descrizione in un <p>.
    """
    url = "https://www.clicksicilia.com/santuari_eremi/santuari-celebri.php"
    headers = {'User-Agent': 'Mozilla/5.0'}
    results = []
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        items = soup.find_all("div", class_="santuario-list-item")
        for item in items:
            title_tag = item.find("h3")
            name = title_tag.text.strip() if title_tag else "N/D"
            desc_tag = item.find("p")
            description = desc_tag.text.strip() if desc_tag else ""
            data = {
                "name": name,
                "description": description,
                "region": "Sicilia",
                "province": "",
                "city": "",
                "lat": "",
                "lon": "",
                "type": "Santuario/Eremo",
                "source": "clicksicilia.com",
                "url": url
            }
            results.append(data)
    except Exception as e:
        logging.error(f"Errore nello scraping di ClickSicilia: {e}")
    return results

def get_monuments_from_tripadvisor() -> list:
    """
    Scraping dal sito TripAdvisor per attrazioni in Sicilia.
    Nota: TripAdvisor tende a bloccare richieste dirette; qui si tenta con un header user-agent aggiornato.
    """
    url = "https://www.tripadvisor.it/Attractions-g187886-Activities-c47-t175-Sicily.html"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'}
    results = []
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        # Ipotizziamo che le attrazioni siano in <div class="attraction_element">
        items = soup.find_all("div", class_="attraction_element")
        for item in items:
            title_tag = item.find("div", class_="listing_title")
            name = title_tag.text.strip() if title_tag else "N/D"
            desc = item.get_text(separator=" ", strip=True)
            data = {
                "name": name,
                "description": desc,
                "region": "Sicilia",
                "province": "",
                "city": "",
                "lat": "",
                "lon": "",
                "type": "Attrazione turistica",
                "source": "tripadvisor.it",
                "url": url
            }
            results.append(data)
    except Exception as e:
        logging.error(f"Errore nello scraping di TripAdvisor: {e}")
    return results

def get_monuments_from_eremos() -> list:
    """
    Scraping dal sito Eremos per santuari ed eremi in Sicilia.
    Ipotizziamo che gli elementi siano contenuti in <div class="item"> e il titolo in <h2>.
    """
    url = "https://eremos.eu/index.php/sicilia/"
    headers = {'User-Agent': 'Mozilla/5.0'}
    results = []
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        items = soup.find_all("div", class_="item")
        for item in items:
            title_tag = item.find("h2")
            name = title_tag.text.strip() if title_tag else "N/D"
            desc_tag = item.find("div", class_="description")
            description = desc_tag.text.strip() if desc_tag else ""
            data = {
                "name": name,
                "description": description,
                "region": "Sicilia",
                "province": "",
                "city": "",
                "lat": "",
                "lon": "",
                "type": "Santuario/Eremo",
                "source": "eremos.eu",
                "url": url
            }
            results.append(data)
    except Exception as e:
        logging.error(f"Errore nello scraping di Eremos: {e}")
    return results

########################################
# UNIONE DELLE FONTI E SALVATAGGIO CSV
########################################

def classify_topic(text: str) -> str:
    text_lower = text.lower()
    if "storia" in text_lower:
        return "storia"
    elif "arte" in text_lower:
        return "arte"
    elif "architettura" in text_lower:
        return "architettura"
    elif "cultura" in text_lower:
        return "cultura"
    else:
        return "generale"

def save_results_csv(monuments: list, csv_filename: str = "monumenti_dettagliati.csv"):
    if not monuments:
        logging.warning("Nessun monumento da salvare.")
        return
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['name', 'description', 'region', 'province', 'city', 'lat', 'lon', 'type', 'source', 'url', 'topic']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for m in monuments:
            if not m.get("topic"):
                m["topic"] = classify_topic(m.get("description", ""))
            writer.writerow(m)
    logging.info(f"File CSV salvato in {csv_filename}")

########################################
# MAIN: RACCOLTA DATI E SALVATAGGIO CSV COMPLETO
########################################

if __name__ == "__main__":
    logging.info("Avvio dello scraping delle Cattedrali della Sicilia da Wikipedia...")
    wiki_monuments = get_all_monuments()
    logging.info(f"Trovati {len(wiki_monuments)} monumenti da Wikipedia.")

    logging.info("Avvio dello scraping dal sito Cultura.gov.it...")
    cultura_monuments = get_monuments_from_cultura()
    logging.info(f"Trovati {len(cultura_monuments)} monumenti da Cultura.gov.it.")

    logging.info("Avvio dello scraping dal sito ClickSicilia...")
    click_monuments = get_monuments_from_clicksicilia()
    logging.info(f"Trovati {len(click_monuments)} monumenti da ClickSicilia.")

    logging.info("Avvio dello scraping dal sito TripAdvisor...")
    trip_monuments = get_monuments_from_tripadvisor()
    logging.info(f"Trovati {len(trip_monuments)} monumenti da TripAdvisor.")

    logging.info("Avvio dello scraping dal sito Eremos...")
    eremos_monuments = get_monuments_from_eremos()
    logging.info(f"Trovati {len(eremos_monuments)} monumenti da Eremos.")

    # Unisci tutte le fonti (sia chiese che cattedrali)
    all_monuments = wiki_monuments + cultura_monuments + click_monuments + trip_monuments + eremos_monuments
    logging.info(f"Totale monumenti combinati: {len(all_monuments)}")
    save_results_csv(all_monuments)

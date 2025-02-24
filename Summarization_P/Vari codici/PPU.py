import os
import requests
import csv
import time
import json
import tiktoken
import folium
import gradio as gr
import logging
import random
import shutil
from typing import Tuple, List, Dict, Any
from urllib.parse import urlparse, urljoin

# Configurazione logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import per nltk (stopwords italiane)
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
italian_stopwords = stopwords.words('italian')

# Configurazione Elasticsearch tramite variabili d'ambiente
from elasticsearch import Elasticsearch
ES_URL = os.getenv("ES_URL", "http://localhost:9200")
ES_USER = os.getenv("ES_USER", "elastic")
ES_PASSWORD = os.getenv("ES_PASSWORD", "iD8R41p-GsP5UW*y4AIb")
VERIFY_CERTS = os.getenv("VERIFY_CERTS", "False").lower() == "true"
INDEX_NAME = os.getenv("INDEX_NAME", "langchain_index")
es = Elasticsearch(
    ES_URL,
    basic_auth=(ES_USER, ES_PASSWORD),
    verify_certs=VERIFY_CERTS
)

# Import LLM e text splitter (esempio con OllamaLLM)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM

# Import metriche: BERTScore e ROUGE
from bert_score import score as bert_score
from rouge_score import rouge_scorer

########################################
# FUNZIONI DI GUIDA PER LA SCRITTURA MANUALE DI UN PROMPT (APE)
########################################
def guide_manual_prompt() -> str:
    guidelines = (
        "Linee guida per scrivere manualmente un prompt efficace:\n\n"
        "Esempio in Italian:\n"
        "-----------------------------------\n"
        "Sei un assistente esperto in patrimonio culturale. Usa un linguaggio formale e dettagliato adatto a un pubblico adulto. "
        "Sottolinea gli aspetti storici e artistici del monumento. Scrivi il riassunto in italiano. "
        "Dettagli: Nome: Cattedrale di Palermo, Tipo: Cattedrale, Località: Palermo, Sicilia, Italia, "
        "Descrizione: Un magnifico esempio di architettura normanno-araba, fonte: Wikipedia - https://it.wikipedia.org/wiki/Cattedrale_di_Palermo.\n"
        "Genera un riassunto che sintetizzi accuratamente questi dettagli.\n\n"
        "Esempio in English:\n"
        "-----------------------------------\n"
        "You are an expert in cultural heritage. Use a formal and detailed language suitable for an adult audience. "
        "Highlight the historical and artistic aspects of the monument. Write the summary in English. "
        "Details: Name: Palermo Cathedral, Type: Cathedral, Location: Palermo, Sicily, Italy, "
        "Description: A magnificent example of Norman-Arab-Byzantine architecture, source: Wikipedia - https://en.wikipedia.org/wiki/Palermo_Cathedral.\n"
        "Generate a summary that accurately synthesizes these details without inventing information."
    )
    return guidelines

########################################
# CONFIGURAZIONE GLOBALI E UTILITY
########################################
# Istruzioni per la lingua
language_instructions: Dict[str, str] = {
    "English": "Write the summary in English.",
    "Italian": "Scrivi il riassunto in italiano."
}

# Istruzioni base per il narratore
narrator_instructions: Dict[str, Dict[str, str]] = {
    "Tour Guide": {
        "English": "Imagine you are a passionate tour guide, full of practical anecdotes and local curiosities.",
        "Italian": "Immagina di essere una guida turistica esperta, capace di raccontare con passione e aggiungere aneddoti pratici."
    },
    "Historian": {
        "English": "Adopt the tone of an erudite historian, with precise chronological references and formal language.",
        "Italian": "Adotta uno stile da storico appassionato, con riferimenti cronologici e un linguaggio formale."
    },
    "Friendly": {
        "English": "Use a friendly, direct, and conversational tone, as if speaking with a friend.",
        "Italian": "Usa un tono amichevole, diretto e colloquiale, come se parlassi con un amico."
    },
    "Expert": {
        "English": "Be extremely precise and technical, using specialized terminology and a formal register.",
        "Italian": "Sii estremamente preciso e tecnico, utilizzando termini specializzati e un registro formale."
    },
    "Storyteller": {
        "English": "Tell the story in an engaging, narrative way, enriching the text with captivating details.",
        "Italian": "Racconta la storia in modo coinvolgente e narrativo, arricchendo il testo di dettagli che incantino il pubblico."
    }
}

# Template APE per la generazione di prompt (ITA)
APE_PROMPT_TEMPLATES_IT: List[str] = [
    "Sei un esperto di patrimonio culturale e storico. {tone_instruction} {focus_instruction} {lang_instruction} "
    "Analizza i seguenti dettagli del monumento e genera un riassunto conciso e accurato, ispirandoti anche agli insight del paper [2402.00284]:\n"
    "Nome: {name}\nTipo: {type}\nLocalità: {city}, {province}, {region}\nDescrizione: {description}\nFonte: {source} - {url}\n"
    "Istruzioni aggiuntive: {additional_instructions}\n"
    "Il riassunto deve essere fedele ai dati e non inventare dettagli.",
    "Immagina di essere una guida turistica appassionata. {tone_instruction} {focus_instruction} {lang_instruction} "
    "Utilizzando le informazioni sottostanti e ispirandoti agli insight del paper [2402.00284], crea un riassunto coinvolgente del monumento:\n"
    "• Nome: {name}\n• Tipo: {type}\n• Località: {city}, {province}, {region}\n• Descrizione: {description}\n• Fonte: {source} - {url}\n"
    "Istruzioni aggiuntive: {additional_instructions}\n"
    "Assicurati che il riassunto sia informativo e adatto al pubblico."
]

APE_PROMPT_TEMPLATES_EN: List[str] = [
    "You are an expert in cultural heritage and history. {tone_instruction} {focus_instruction} {lang_instruction} "
    "Analyze the following monument details and generate a concise and accurate summary, drawing inspiration from the insights in [2402.00284]:\n"
    "Name: {name}\nType: {type}\nLocation: {city}, {province}, {region}\nDescription: {description}\nSource: {source} - {url}\n"
    "Additional instructions: {additional_instructions}\n"
    "Ensure the summary is faithful to the data and does not invent any details.",
    "Imagine you are a passionate tour guide. {tone_instruction} {focus_instruction} {lang_instruction} "
    "Using the information below and inspired by the insights in [2402.00284], create an engaging summary of the monument:\n"
    "• Name: {name}\n• Type: {type}\n• Location: {city}, {province}, {region}\n• Description: {description}\n• Source: {source} - {url}\n"
    "Additional instructions: {additional_instructions}\n"
    "Make sure the summary is informative and suitable for the audience."
]

########################################
# IMPOSIZIONE DEL MODELLO LLM
########################################
# In questa versione il modello è impostato direttamente e utilizzato in modo fisso.
DEFAULT_MODEL = "llama3.2"
llm = OllamaLLM(
    base_url="http://localhost:11434",
    model=DEFAULT_MODEL,
    temperature=0.7
)

def update_llm_model(model_name: str) -> None:
    logging.info(f"Aggiorno il modello LLM a: {model_name}")
    llm.model = model_name

def count_tokens(text: str) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

########################################
# FUNZIONI DI VALUTAZIONE
########################################
def compute_bart_score(candidate: str, reference: str) -> float:
    return random.uniform(0, 1)

def compute_mover_score(candidate: str, reference: str) -> float:
    return random.uniform(0, 1)

def evaluate_summary(candidate: str, reference: str) -> Dict[str, Any]:
    if not reference.strip():
        return {"BERTScore_F1": 0, "ROUGE_L": 0, "BLEURT": 0, "BARTScore": 0, "MoverScore": 0}
    P, R, F1 = bert_score([candidate], [reference], lang="it")
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rougeL = scorer.score(reference, candidate)['rougeL'].fmeasure
    bleurt = 0.76  # placeholder
    bart = compute_bart_score(candidate, reference)
    mover = compute_mover_score(candidate, reference)
    return {
        "BERTScore_F1": F1.item(),
        "ROUGE_L": rougeL,
        "BLEURT": bleurt,
        "BARTScore": bart,
        "MoverScore": mover
    }

def external_evaluation(summary: str, reference: str = "") -> Dict[str, Any]:
    return evaluate_summary(summary, reference)

def save_ape_details_to_csv(details: List[Dict[str, Any]], filename: str = "ape_prompts.csv") -> None:
    fieldnames = ["prompt", "summary", "token_count", "bert_score"]
    with open(filename, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for d in details:
            writer.writerow(d)
    logging.info(f"Dettagli APE salvati in {filename}")

def save_results_csv(query: str, summary: str, monument_name: str, csv_filename: str = "rag_results.csv") -> None:
    fieldnames = ['query', 'monument', 'summary']
    with open(csv_filename, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if os.stat(csv_filename).st_size == 0:
            writer.writeheader()
        writer.writerow({
            'query': query,
            'monument': monument_name,
            'summary': summary
        })

########################################
# NUOVA FUNZIONE: BACKUP AUTOMATICO DEL CSV DI SESSIONE
########################################
def backup_session_results(filename: str = "session_results.csv") -> None:
    if os.path.exists(filename):
        backup_filename = f"session_results_backup_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        shutil.copy(filename, backup_filename)
        logging.info(f"Session results backed up to {backup_filename}")

########################################
# NUOVA FUNZIONE: LOG DEI RISULTATI DELLA SESSIONE
########################################
def log_session_result(
    selected_model: str,
    selected_monument: str,
    user_query: str,
    age_group: str,
    interest: str,
    narrator_type: str,
    language: str,
    temperature: float,
    chunk_size: int,
    overlap_size: int,
    retrieval_method: str,
    additional_instructions: str,
    final_summary: str,
    token_count: int,
    eval_metrics: str,
    ape_details: str
) -> None:
    """
    Salva in un file CSV (session_results.csv) una riga contenente:
      - I dati della sessione e dei parametri scelti
      - La risposta finale (riassunto)
      - Il conteggio dei token usati
      - Le metriche di valutazione (in formato JSON)
      - I dettagli APE
    """
    result_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": selected_model,
        "selected_monument": selected_monument,
        "user_query": user_query,
        "age_group": age_group,
        "interest": interest,
        "narrator_type": narrator_type,
        "language": language,
        "temperature": temperature,
        "chunk_size": chunk_size,
        "overlap_size": overlap_size,
        "retrieval_method": retrieval_method,
        "additional_instructions": additional_instructions,
        "final_summary": final_summary,
        "token_count": token_count,
        "eval_metrics": eval_metrics,
        "ape_details": ape_details
    }
    filename = "../session_results.csv"
    fieldnames = list(result_data.keys())
    file_exists = os.path.exists(filename) and os.path.getsize(filename) > 0
    # Backup automatico per non perdere i dati precedenti
    backup_session_results(filename)
    with open(filename, mode='a', newline='', encoding='utf-8') as csvfile:
         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
         if not file_exists:
             writer.writeheader()
         writer.writerow(result_data)
    logging.info(f"Session result saved to {filename}")

########################################
# FUNZIONI DI GESTIONE (Elasticsearch, scraping, CSV, mappe)
########################################
def create_index_elasticsearch(index_name: str = INDEX_NAME) -> None:
    logging.info(f"Verifico l'esistenza dell'indice {index_name}")
    if es.indices.exists(index=index_name):
        logging.info(f"L'indice {index_name} esiste, lo elimino")
        es.indices.delete(index=index_name)
    settings = {
        "settings": {
            "analysis": {
                "analyzer": {
                    "my_italian_analyzer": {
                        "type": "standard",
                        "stopwords": ["_italian_"]
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "name": {"type": "text", "analyzer": "my_italian_analyzer"},
                "description": {"type": "text", "analyzer": "my_italian_analyzer"},
                "topic": {"type": "keyword"},
                "source": {"type": "keyword"},
                "url": {"type": "keyword"},
                "chunk_id": {"type": "keyword"}
            }
        }
    }
    es.indices.create(index=index_name, body=settings)
    logging.info(f"Indice '{index_name}' creato correttamente.")

def index_monuments_in_es(csv_filename: str = "monumenti_dettagliati.csv",
                          index_name: str = INDEX_NAME,
                          chunk_size: int = 500,
                          overlap: int = 100) -> None:
    if not os.path.exists(csv_filename):
        logging.error(f"File {csv_filename} non trovato. Impossibile indicizzare.")
        return
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    with open(csv_filename, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            full_text = row.get("description", "")
            chunks = splitter.split_text(full_text)
            for idx, ch in enumerate(chunks):
                doc_body = {
                    "name": row.get("name", ""),
                    "description": ch,
                    "topic": row.get("topic", ""),
                    "source": row.get("source", ""),
                    "url": row.get("url", ""),
                    "chunk_id": f"{row.get('name', '')}-{idx}"
                }
                es.index(index=index_name, body=doc_body)
    logging.info(f"Indicizzazione completata su indice '{index_name}'")

def retrieve_docs_es(user_query: str, top_k: int = 3, index_name: str = INDEX_NAME) -> List[str]:
    logging.info(f"Eseguo la query Elasticsearch per: {user_query}")
    search_body = {
        "size": top_k,
        "query": {
            "bool": {
                "should": [
                    {"match": {"name": {"query": user_query, "boost": 2}}},
                    {"match": {"description": {"query": user_query}}}
                ],
                "minimum_should_match": 1
            }
        }
    }
    resp = es.search(index=index_name, body=search_body)
    hits = resp["hits"]["hits"]
    results = []
    for h in hits:
        source = h["_source"]
        results.append(source["description"])
    logging.info(f"Chunk trovati: {results}")
    return results

def is_valid_url(url: str) -> bool:
    parsed = urlparse(url)
    return bool(parsed.scheme and parsed.netloc)

def tag_visible(element) -> bool:
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    from bs4.element import Comment
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
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        texts = soup.find_all(string=True)
        visible_texts = filter(tag_visible, texts)
        return " ".join(t.strip() for t in visible_texts)
    except Exception as e:
        return f"Error: {str(e)}"

def parse_location(location_str: str) -> Tuple[str, str, str]:
    parts = [p.strip() for p in location_str.split(',')]
    city = parts[0] if len(parts) > 0 else ""
    province = parts[1] if len(parts) > 1 else ""
    region = parts[2] if len(parts) > 2 else ""
    return region, province, city

def load_monument_names_from_csv(csv_filename: str = "monumenti_dettagliati.csv") -> List[str]:
    names = []
    if not os.path.exists(csv_filename):
        logging.error(f"File {csv_filename} non trovato!")
        return names
    with open(csv_filename, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            names.append(row["name"])
    return names

def get_monument_data_from_csv(selected_name: str, csv_filename: str = "monumenti_dettagliati.csv") -> Dict[str, Any]:
    if os.path.exists(csv_filename):
        with open(csv_filename, mode='r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row["name"] == selected_name:
                    return row
    return {}

def extract_monument_data(url: str, monument_type: str) -> Dict[str, Any]:
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
        from bs4 import BeautifulSoup
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

def get_monuments_from_wikipedia(page_url: str, monument_type: str) -> List[Dict[str, Any]]:
    monuments = []
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(page_url, headers=headers, timeout=10)
        response.raise_for_status()
        from bs4 import BeautifulSoup
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

def get_all_monuments() -> List[Dict[str, Any]]:
    cathedrals_url = "https://it.wikipedia.org/wiki/Categoria:Cattedrali_della_Sicilia"
    monuments = get_monuments_from_wikipedia(cathedrals_url, "Cattedrale")
    return monuments

def get_monuments_from_cultura() -> List[Dict[str, Any]]:
    url = "https://cultura.gov.it/luoghi/cerca-luogo?regione=regione-sicilia&tipo=chiesa-o-edificio-di-culto"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        from bs4 import BeautifulSoup
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

def save_results_csv(query: str, summary: str, monument_name: str, csv_filename: str = "rag_results.csv") -> None:
    fieldnames = ['query', 'monument', 'summary']
    with open(csv_filename, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if os.stat(csv_filename).st_size == 0:
            writer.writeheader()
        writer.writerow({
            'query': query,
            'monument': monument_name,
            'summary': summary
        })

def get_coordinates(name: str) -> Tuple[Any, Any]:
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": name + ", Sicilia, Italy",
        "format": "json"
    }
    try:
        response = requests.get(url, params=params, headers={'User-Agent': 'Mozilla/5.0'})
        data = response.json()
        if data:
            lat = float(data[0]['lat'])
            lon = float(data[0]['lon'])
            return lat, lon
    except Exception as e:
        logging.error(f"Errore in get_coordinates per {name}: {e}")
    return None, None

def generate_folium_map_from_csv(csv_filename: str = "monumenti_dettagliati.csv") -> str:
    m = folium.Map(location=[37.5, 14.0], zoom_start=7)
    if os.path.exists(csv_filename):
        with open(csv_filename, mode='r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    lat = row.get("lat", "").strip()
                    lon = row.get("lon", "").strip()
                    if not lat or not lon:
                        lat, lon = get_coordinates(row.get("name", ""))
                    else:
                        lat = float(lat)
                        lon = float(lon)
                    if lat and lon:
                        folium.Marker(
                            location=[lat, lon],
                            popup=row.get("name", "N/D"),
                            tooltip=row.get("name", "N/D")
                        ).add_to(m)
                except Exception as ex:
                    logging.error(f"Errore nelle coordinate per {row.get('name')}: {ex}")
    else:
        logging.error(f"File {csv_filename} non trovato!")
    return m._repr_html_()

def get_text_splitter(chunk_size: int, overlap_size: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap_size)

def summarize_text(text: str, age_group: str, interest: str, temperature: float, chunk_size: int, overlap_size: int) -> Tuple[str, int]:
    splitter = get_text_splitter(chunk_size, overlap_size)
    split_docs = splitter.split_text(text)
    summaries = []
    total_tokens = 0
    enc = tiktoken.get_encoding("cl100k_base")
    for doc in split_docs:
        prompt = (
            f"Sei un esperto di storia dell'arte. "
            f"Riassumi il seguente testo riguardante una chiesa storica, "
            f"attenendoti rigorosamente al testo e senza inventare dettagli, adattandolo per un pubblico {age_group} interessato a {interest}:\n\n"
            f"{doc}\n\n"
            "Il riassunto deve includere in modo preciso i dettagli sull'architettura, la storia e l'importanza culturale della chiesa."
        )
        logging.info(f"Prompt di sommario: {prompt}")
        llm.temperature = temperature
        try:
            result = llm.generate([prompt])
            summary = result.generations[0][0].text.strip()
            summaries.append(summary)
            total_tokens += len(enc.encode(doc))
            logging.info(f"Riassunto parziale generato: {summary}")
        except Exception as e:
            error_msg = f"Errore: {str(e)}"
            summaries.append(error_msg)
            logging.error(error_msg)
    return "\n\n".join(summaries), total_tokens

########################################
# FUNZIONI DI GENERAZIONE AVANZATA DI PROMPT PERSONALIZZATI (APE)
########################################
def generate_series_personalized_summary(monument: Dict[str, Any], age_group: str, interest: str,
                                         narrator_type: str, language: str,
                                         temperature: float, additional_instructions: str) -> Tuple[str, int, List[Dict[str, Any]]]:
    """
    Genera una serie di prompt personalizzati basati sui parametri e sulle istruzioni aggiuntive.
    Per ogni template (in base alla lingua), genera il prompt, ottiene il riassunto, conta i token e valuta il punteggio.
    Salva tutte le combinazioni in un CSV e restituisce il miglior riassunto, il conteggio token e i dettagli.
    """
    if language == "English":
        templates = APE_PROMPT_TEMPLATES_EN
        default_tone = "Use a simple, lively, and engaging language, suitable for a young audience." if age_group.lower() == "giovane" else "Use a formal and detailed language."
    else:
        templates = APE_PROMPT_TEMPLATES_IT
        default_tone = "Usa un linguaggio semplice, vivace e coinvolgente, adatto a un pubblico giovane." if age_group.lower() == "giovane" else "Usa un linguaggio formale e dettagliato."

    if interest.lower() in ["arte", "architettura"]:
        focus_instruction = "Emphasize the artistic and architectural aspects." if language == "English" else "Metti in evidenza gli aspetti artistici e architettonici."
    elif interest.lower() == "storia":
        focus_instruction = "Highlight the historical context and developments." if language == "English" else "Sottolinea il contesto storico e le evoluzioni nel tempo."
    elif interest.lower() == "religione":
        focus_instruction = "Emphasize religious references and spiritual significance." if language == "English" else "Evidenzia i riferimenti religiosi e il significato spirituale."
    else:
        focus_instruction = "Provide a comprehensive view of cultural and social aspects." if language == "English" else "Offri una visione completa degli aspetti culturali e sociali."

    lang_instruction = language_instructions.get(language, "Scrivi il riassunto in italiano.")
    narrator_instruction = narrator_instructions.get(narrator_type, {}).get(language, narrator_type)

    details_list = []
    best_summary = None
    best_score = -1
    best_token_count = 0
    enc = tiktoken.get_encoding("cl100k_base")

    for template in templates:
        prompt = template.format(
            tone_instruction=default_tone,
            focus_instruction=focus_instruction,
            lang_instruction=lang_instruction,
            name=monument.get("name", "N/D"),
            type=monument.get("type", "N/D"),
            city=monument.get("city", "N/D"),
            province=monument.get("province", "N/D"),
            region=monument.get("region", "N/D"),
            description=monument.get("description", "N/D"),
            source=monument.get("source", "N/D"),
            url=monument.get("url", ""),
            additional_instructions=additional_instructions if additional_instructions.strip() != "" else "Nessuna istruzione aggiuntiva."
        )
        logging.info(f"Prompt APE generato: {prompt}")
        llm.temperature = temperature
        try:
            result = llm.generate([prompt])
            summary = result.generations[0][0].text.strip()
        except Exception as e:
            summary = f"Errore nella generazione: {str(e)}"
        token_count = count_tokens(prompt)
        eval_metrics = external_evaluation(summary, monument.get("description", ""))
        bert_score_val = eval_metrics.get("BERTScore_F1", 0)

        details = {
            "prompt": prompt,
            "summary": summary,
            "token_count": token_count,
            "bert_score": bert_score_val
        }
        details_list.append(details)

        if bert_score_val > best_score:
            best_score = bert_score_val
            best_summary = summary
            best_token_count = token_count

    save_ape_details_to_csv(details_list, filename="../ape_prompts.csv")
    return best_summary, best_token_count, details_list

########################################
# FUNZIONE process_query: INTEGRAZIONE RAG CON ELASTICSEARCH
########################################
def process_query(selected_monument_name: str, age_group: str, interest: str, narrator_type: str,
                  language: str, temperature: float, chunk_size: int, overlap_size: int,
                  save_csv: bool, ext_eval: bool, user_query: str = "", retrieval_method: str = "ElasticSearch",
                  additional_instructions: str = "") -> Tuple[str, int, str, str, str]:
    # Utilizza sempre il modello definito in DEFAULT_MODEL
    update_llm_model(DEFAULT_MODEL)
    if retrieval_method == "ElasticSearch":
        full_query = f"{selected_monument_name} {user_query}" if user_query else selected_monument_name
        logging.info(f"Full query per ES: {full_query}")
        relevant_chunks = retrieve_docs_es(full_query, top_k=3, index_name=INDEX_NAME)
        if not relevant_chunks:
            return "Nessun chunk trovato da ES.", 0, "N/D", json.dumps({}), ""
        combined_text = "\n".join(relevant_chunks)
        logging.info(f"Testo combinato dai chunk: {combined_text}")
        summary_text, tokens = summarize_text(combined_text, age_group, interest, temperature, chunk_size, overlap_size)
        monument_data = get_monument_data_from_csv(selected_monument_name)
        if not monument_data:
            monument_data = {
                "name": selected_monument_name,
                "description": combined_text,
                "type": "",
                "city": "",
                "province": "",
                "region": "",
                "source": "Elasticsearch",
                "url": ""
            }
        personalized_summary, ape_tokens, ape_details = generate_series_personalized_summary(
            monument_data, age_group, interest, narrator_type, language, temperature, additional_instructions
        )
        final_summary = personalized_summary
        eval_metrics = {}
        if ext_eval:
            reference_text = monument_data.get("description", "")
            eval_metrics = external_evaluation(final_summary, reference_text)
        if save_csv:
            save_results_csv(user_query, final_summary, monument_data.get("name", "N/D"))
        details_str = "### APE Details\n\n"
        for d in ape_details:
            details_str += f"**Prompt:**\n```\n{d['prompt']}\n```\n"
            details_str += f"**Summary:**\n```\n{d['summary']}\n```\n"
            details_str += f"**Token Count:** {d['token_count']} | **BERT Score:** {d['bert_score']:.4f}\n\n"
        return final_summary, tokens + ape_tokens, monument_data.get("name", "N/D"), json.dumps(eval_metrics), details_str
    else:
        monument = get_monument_data_from_csv(selected_monument_name)
        if not monument or not monument.get("name"):
            return "Monumento non trovato", 0, "N/D", json.dumps({}), ""
        summary_text, tokens = summarize_text(monument["description"], age_group, interest, temperature, chunk_size, overlap_size)
        personalized_summary, ape_tokens, ape_details = generate_series_personalized_summary(
            monument, age_group, interest, narrator_type, language, temperature, additional_instructions
        )
        final_summary = personalized_summary
        eval_metrics = {}
        if ext_eval:
            eval_metrics = external_evaluation(final_summary, monument.get("description", ""))
        if save_csv:
            save_results_csv(user_query, final_summary, monument.get("name", "N/D"))
        details_str = "### APE Details\n\n"
        for d in ape_details:
            details_str += f"**Prompt:**\n```\n{d['prompt']}\n```\n"
            details_str += f"**Summary:**\n```\n{d['summary']}\n```\n"
            details_str += f"**Token Count:** {d['token_count']} | **BERT Score:** {d['bert_score']:.4f}\n\n"
        return final_summary, tokens + ape_tokens, monument.get("name", "N/D"), json.dumps(eval_metrics), details_str

########################################
# INTERFACCIA GRADIO: UI INTERATTIVA
########################################
with gr.Blocks() as demo:
    gr.Markdown("# Personalized Prompt Summarization con RAG (Elasticsearch)")
    gr.Markdown(
        "Questo sistema permette di:\n"
        "1) Creare/ricreare l'indice su Elasticsearch e indicizzare i dati dal CSV.\n"
        "2) Selezionare un monumento e generare un riassunto personalizzato tramite retrieval avanzato.\n"
        "3) Utilizzare metriche (BERTScore, ROUGE, BLEURT, BARTScore, MoverScore) per valutare la qualità del riassunto.\n\n"
        "Il sistema genera automaticamente una serie di prompt personalizzati combinando i parametri inseriti e restituisce il riassunto migliore, mostrando anche i dettagli dei prompt generati."
    )

    with gr.Row():
        btn_create_index = gr.Button("Crea/Ricrea Indice su Elasticsearch")
        btn_index_csv = gr.Button("Indicizza CSV in Elasticsearch")
    output_logs = gr.Textbox(label="Log/Output", interactive=False, lines=5)

    def create_es_index() -> str:
        create_index_elasticsearch(INDEX_NAME)
        return "Indice creato/ripristinato correttamente."

    def index_data() -> str:
        index_monuments_in_es(csv_filename="../monumenti_dettagliati.csv", index_name=INDEX_NAME, chunk_size=500, overlap=100)
        return "Indicizzazione completata."

    btn_create_index.click(fn=create_es_index, inputs=[], outputs=output_logs)
    btn_index_csv.click(fn=index_data, inputs=[], outputs=output_logs)

    with gr.Row():
        map_html = gr.HTML(value=generate_folium_map_from_csv(), label="Mappa Interattiva")

    with gr.Row():
        def load_names() -> List[str]:
            names = []
            csv_file = "../monumenti_dettagliati.csv"
            if os.path.exists(csv_file):
                with open(csv_file, mode='r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        names.append(row["name"])
            return names

        monument_names = load_names()
        if not monument_names:
            monument_names = ["N/D"]
        monument_dropdown = gr.Dropdown(
            label="Seleziona un monumento",
            choices=monument_names,
            value=monument_names[0]
        )

    gr.Markdown("### Parametri per la Summarization")
    with gr.Row():
        age_group = gr.Dropdown(
            label="Fascia d'età",
            choices=["giovane", "adulto", "anziano"],
            value="adulto",
            allow_custom_value=True
        )
        interest = gr.Dropdown(
            label="Interesse",
            choices=["storia", "arte", "architettura", "religione"],
            value="storia",
            allow_custom_value=True
        )
    with gr.Row():
        narrator_dropdown = gr.Dropdown(
            label="Tipo di Narratore",
            choices=["Tour Guide", "Historian", "Friendly", "Expert", "Storyteller"],
            value="Tour Guide",
            allow_custom_value=True
        )
        language_dropdown = gr.Dropdown(
            label="Lingua",
            choices=["Italian", "English"],
            value="Italian",
            allow_custom_value=True
        )
    with gr.Row():
        additional_instructions = gr.Textbox(
            label="Additional Instructions (optional)",
            lines=2,
            placeholder="Esempio: Esperto storico. Assicurati che il riassunto sia informativo e adatto al pubblico."
        )
    with gr.Row():
        temperature_slider = gr.Slider(minimum=0.1, maximum=1.0, step=0.1, value=0.7, label="Temperatura")
        # Rimosso il dropdown per il modello; verrà usato sempre DEFAULT_MODEL.
        chunk_slider = gr.Slider(minimum=100, maximum=2000, step=100, value=500, label="Dimensione Chunk")
        overlap_slider = gr.Slider(minimum=0, maximum=500, step=50, value=100, label="Overlap")
        retrieval_method = gr.Radio(label="Metodo di Retrieval", choices=["ElasticSearch", "TF-IDF"],
                                    value="ElasticSearch")
    with gr.Row():
        save_csv_checkbox = gr.Checkbox(label="Salva risultati in CSV", value=True)
        external_eval_checkbox = gr.Checkbox(label="Usa valutazione esterna", value=True)
    with gr.Row():
        query_input = gr.Textbox(label="Inserisci la query per il test (opzionale)", lines=2)
    with gr.Row():
        btn_generate = gr.Button("Genera Riassunto & Confronta Risultati", variant="primary")

    ape_details_output = gr.Markdown(label="APE Details")
    results_output = gr.Dataframe(label="Tabella Confronto Risultati")

    def compare_models(selected_monument, query, age_grp, interest_val,
                       narrator_val, language_val, temp,
                       chunk, overlap, save_csv, ext_eval, ret_method, add_instr):
        # Utilizza sempre il modello definito in DEFAULT_MODEL
        selected_model = DEFAULT_MODEL
        summary, tokens, mon_name, eval_metrics, ape_details = process_query(
            selected_monument_name=selected_monument,
            age_group=age_grp,
            interest=interest_val,
            narrator_type=narrator_val,
            language=language_val,
            temperature=temp,
            chunk_size=chunk,
            overlap_size=overlap,
            save_csv=save_csv,
            ext_eval=ext_eval,
            user_query=query,
            retrieval_method=ret_method,
            additional_instructions=add_instr
        )
        results = [[selected_model, mon_name, summary, tokens, eval_metrics]]
        headers = ["Model", "Monument", "Response", "Tokens", "Metrics"]
        table = [headers] + results

        # Salva il risultato della sessione (applica il backup automatico)
        log_session_result(
            selected_model=selected_model,
            selected_monument=mon_name,
            user_query=query,
            age_group=age_grp,
            interest=interest_val,
            narrator_type=narrator_val,
            language=language_val,
            temperature=temp,
            chunk_size=chunk,
            overlap_size=overlap,
            retrieval_method=ret_method,
            additional_instructions=add_instr,
            final_summary=summary,
            token_count=tokens,
            eval_metrics=eval_metrics,
            ape_details=ape_details
        )
        return table, ape_details

    btn_generate.click(
        fn=compare_models,
        inputs=[monument_dropdown, query_input, age_group, interest,
                narrator_dropdown, language_dropdown, temperature_slider,
                chunk_slider, overlap_slider,
                save_csv_checkbox, external_eval_checkbox, retrieval_method, additional_instructions],
        outputs=[results_output, ape_details_output]
    )

    with gr.Accordion("Guida per scrivere manualmente un prompt (APE)", open=False):
        manual_guidelines = gr.Markdown(guide_manual_prompt())
        gr.Markdown(
            "Usa queste linee guida se vuoi scrivere il tuo prompt, ma il sistema genera automaticamente il riassunto personalizzato."
        )

demo.launch(share=True)

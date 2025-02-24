import requests
from bs4 import BeautifulSoup
import csv
import re
import time

def clean_text(txt):
    """Rimuove spazi e ritorna il testo pulito."""
    return re.sub(r'\s+', ' ', txt).strip()

def get_wikipedia_extract(title):
    """
    Cerca la pagina Wikipedia utilizzando l'azione opensearch per ottenere suggerimenti,
    quindi restituisce l'estratto della prima pagina trovata.
    """
    api_url = "https://it.wikipedia.org/w/api.php"
    params = {
        "action": "opensearch",
        "format": "json",
        "search": title,
        "limit": 1,
        "namespace": 0
    }
    try:
        resp = requests.get(api_url, params=params, timeout=20)
        if resp.status_code == 200:
            data = resp.json()
            # data[1] contiene i titoli trovati
            if data and len(data) > 1 and data[1]:
                found_title = data[1][0]
                # Richiedi l'estratto della pagina trovata
                params_extract = {
                    "action": "query",
                    "format": "json",
                    "prop": "extracts",
                    "exintro": True,
                    "explaintext": True,
                    "redirects": True,
                    "titles": found_title
                }
                resp_extract = requests.get(api_url, params=params_extract, timeout=20)
                if resp_extract.status_code == 200:
                    extract_data = resp_extract.json()
                    pages = extract_data.get("query", {}).get("pages", {})
                    for page_id, page in pages.items():
                        # Se page_id è -1, significa che la pagina non è stata trovata
                        if page_id == "-1":
                            return ""
                        return clean_text(page.get("extract", ""))
    except Exception as e:
        print(f"Errore durante la richiesta a Wikipedia per '{title}': {e}")
    return ""

# URL del sito eremos.eu
url = "https://eremos.eu/index.php/sicilia/"

# Effettua la richiesta al sito
response = requests.get(url)
if response.status_code != 200:
    print(f"Errore nella richiesta: {response.status_code}")
    exit(1)

# Parsing dell'HTML con BeautifulSoup
soup = BeautifulSoup(response.text, "html.parser")
entry = soup.find("div", class_="entry-content clearfix")
if not entry:
    print("Non è stato possibile trovare il contenuto principale")
    exit(1)

# Raggruppa gli elementi in blocchi separati da <hr>
blocks = []
current_block = []
for child in entry.children:
    if not hasattr(child, "name"):
        continue
    if child.name == "hr":
        if current_block:
            blocks.append(current_block)
            current_block = []
    elif child.name in ["p", "h3"]:
        text = child.get_text(strip=True)
        if text:
            current_block.append(text)
if current_block:
    blocks.append(current_block)

# Lista per contenere tutti i record estratti
records = []

# Estrae i dati per ciascun blocco
for block in blocks:
    if len(block) < 2:
        continue

    record = {
        "name": "",
        "type": "",
        "location": "",
        "province": "",
        "sub_area": "",
        "description": "",
        "source": "eremos.eu",
        "url": url,
        "topic": "generale",
        "wikipedia_extract": ""
    }

    # Il titolo è la prima riga; se contiene una parte tra parentesi, la consideriamo come "type"
    title_line = block[0]
    m = re.search(r'\(([^)]+)\)', title_line)
    if m:
        record["type"] = clean_text(m.group(1))
        title_line = re.sub(r'\s*\([^)]+\)', '', title_line)
    record["name"] = clean_text(title_line)

    # Cerca la riga che inizia con "Località:" per estrarre location, province e sub_area
    for line in block:
        if line.startswith("Località:"):
            m_loc = re.search(r'Località:\s*(.*?)\s*\(([^)]+)\)', line)
            if m_loc:
                record["location"] = clean_text(m_loc.group(1))
                record["province"] = clean_text(m_loc.group(2))
            m_sub = re.search(r'Sub-area:\s*(.*)', line)
            if m_sub:
                record["sub_area"] = clean_text(m_sub.group(1))
            break

    # La descrizione è composta dalle altre righe, escludendo la riga con "Località:" e il titolo ripetuto
    desc_lines = []
    for line in block:
        if line.startswith("Località:"):
            continue
        if clean_text(line) == record["name"]:
            continue
        desc_lines.append(line)
    record["description"] = clean_text(" ".join(desc_lines))

    records.append(record)

print(f"Totale record estratti da eremos.eu: {len(records)}")
print("Ricerca degli estratti Wikipedia per ciascun record...")

# Per ogni record, cerca l'estratto da Wikipedia
for i, rec in enumerate(records, start=1):
    title = rec["name"]
    # Se il record riguarda una chiesa, cattedrale o eremo, prova ad aggiungere un prefisso se non già presente
    if rec["type"] and ("Chiesa" in rec["type"] or "Cattedrale" in rec["type"] or "Ermo" in rec["type"]):
        if not title.lower().startswith("chiesa") and not title.lower().startswith("cattedrale"):
            title = "Chiesa di " + title
    extract = get_wikipedia_extract(title)
    if extract:
        rec["wikipedia_extract"] = extract
        print(f"[{i}/{len(records)}] '{rec['name']}' -> Trovato extract")
    else:
        print(f"[{i}/{len(records)}] '{rec['name']}' -> Non trovato")
    time.sleep(0.5)  # Piccola pausa per non sovraccaricare l'API

# Salva tutti i record in un file CSV unico
csv_filename = "monumenti_dettagliati.csv"
with open(csv_filename, "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["name", "type", "location", "province", "sub_area", "description", "source", "url", "topic", "wikipedia_extract"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for rec in records:
        writer.writerow(rec)

print(f"Salvato CSV unico: {csv_filename}")

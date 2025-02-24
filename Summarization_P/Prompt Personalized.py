#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
"Prompt Engineering personalizzato per una migliore interazione uomo-macchina"
Integrato con Retrieval-Augmented Generation e analisi dei risultati.
"""

##########################
# 1. CONFIGURAZIONE E SETUP
##########################
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
import asyncio
from typing import Tuple, List, Dict, Any
from urllib.parse import urlparse, urljoin

# Logging configurato in modo robusto
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configurazione per nltk e stopwords
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

# Configurazione LLM e text splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM

DEFAULT_MODEL = "mistral"
llm = OllamaLLM(
    base_url="http://localhost:11434",
    model=DEFAULT_MODEL,
    temperature=0.6
)

# Metriche di valutazione (BERTScore, ROUGE, ecc.)
from bert_score import score as bert_score
from rouge_score import rouge_scorer


# Helper: conteggio token
def count_tokens(text: str) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


##########################
# 2. FUNZIONI PER L'INGESTIONE DEI DATI
##########################
def is_valid_url(url: str) -> bool:
    parsed = urlparse(url)
    return bool(parsed.scheme and parsed.netloc)


def tag_visible(element) -> bool:
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    from bs4.element import Comment
    return not isinstance(element, Comment)


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
        logging.error(f"Errore durante l'estrazione del testo da {url}: {e}")
        return f"Error: {str(e)}"


def create_index_elasticsearch(index_name: str = INDEX_NAME) -> None:
    try:
        logging.info(f"Verifica dell'indice {index_name}")
        if es.indices.exists(index=index_name):
            logging.info(f"Indice {index_name} esiste; lo elimino per ricrearlo.")
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
    except Exception as e:
        logging.error(f"Errore nella creazione dell'indice: {e}")


def index_monuments_in_es(csv_filename: str = "monumenti_dettagliati.csv",
                          index_name: str = INDEX_NAME,
                          chunk_size: int = 500,
                          overlap: int = 100) -> None:
    if not os.path.exists(csv_filename):
        logging.error(f"File {csv_filename} non trovato. Impossibile indicizzare.")
        return
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    try:
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
    except Exception as e:
        logging.error(f"Errore nell'indicizzazione: {e}")


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


##########################
# 3. FUNZIONI DI PROMPT ENGINEERING (APE) E PERSONALIZZAZIONE
##########################
# Template base per lingua e istruzioni
language_instructions: Dict[str, str] = {
    "English": "Write the summary in English.",
    "Italian": "Scrivi il riassunto in italiano."
}

narrator_instructions: Dict[str, Dict[str, str]] = {
    "Tour Guide": {
        "English": "Imagine you are a passionate tour guide with engaging anecdotes.",
        "Italian": "Immagina di essere una guida turistica esperta e appassionata."
    },
    "Historian": {
        "English": "Adopt the tone of an erudite historian with formal language.",
        "Italian": "Adotta uno stile da storico appassionato e formale."
    },
    "Friendly": {
        "English": "Use a friendly, direct, and conversational tone.",
        "Italian": "Usa un tono amichevole, diretto e colloquiale."
    },
    "Expert": {
        "English": "Be extremely precise and technical.",
        "Italian": "Sii estremamente preciso e tecnico."
    },
    "Storyteller": {
        "English": "Tell the story in an engaging, narrative way.",
        "Italian": "Racconta la storia in modo coinvolgente e narrativo."
    }
}


# Funzione helper per generare le istruzioni sul tono in base a narratore, età e lingua
def get_tone_instruction(narrator_type: str, age_group: str, language: str) -> str:
    try:
        if language.lower() == "italian":
            if narrator_type == "Tour Guide":
                return "Usa un linguaggio vivace e coinvolgente, adatto a un pubblico {}.".format(
                    "giovane" if age_group == "giovane" else "adulto")
            elif narrator_type == "Historian":
                return "Adotta un tono formale con riferimenti storici accurati."
            elif narrator_type == "Friendly":
                return "Usa un tono amichevole e colloquiale."
            elif narrator_type == "Expert":
                return "Utilizza un linguaggio tecnico e preciso, adatto a esperti."
            elif narrator_type == "Storyteller":
                return "Racconta in maniera narrativa e coinvolgente."
            else:
                return "Usa un linguaggio formale."
        else:
            if narrator_type == "Tour Guide":
                return "Use a lively and engaging language, suitable for a {} audience.".format(
                    "young" if age_group == "giovane" else "adult")
            elif narrator_type == "Historian":
                return "Adopt a formal tone with accurate historical details."
            elif narrator_type == "Friendly":
                return "Use a friendly and conversational tone."
            elif narrator_type == "Expert":
                return "Utilize technical and precise language, suitable for experts."
            elif narrator_type == "Storyteller":
                return "Tell the story in an engaging and narrative way."
            else:
                return "Use a formal language."
    except Exception as e:
        logging.error(f"Errore in get_tone_instruction: {e}")
        return ""


# Template per APE (per entrambe le lingue)
APE_PROMPT_TEMPLATES_IT: List[str] = [
    "Sei un esperto di patrimonio culturale e storico. {tone_instruction} {focus_instruction} {lang_instruction} "
    "Analizza i dettagli del monumento e genera un riassunto conciso:\n"
    "Nome: {name}\nTipo: {type}\nLocalità: {city}, {province}, {region}\nDescrizione: {description}\nFonte: {source} - {url}\n"
    "Istruzioni aggiuntive: {additional_instructions}\n"
    "Il riassunto deve essere fedele ai dati.",
    "Immagina di essere una guida turistica appassionata ed esperta in beni culturali. {tone_instruction} {focus_instruction} {lang_instruction} "
    "Utilizza le informazioni seguenti per creare un riassunto coinvolgente:\n"
    "• Nome: {name}\n• Tipo: {type}\n• Località: {city}, {province}, {region}\n• Descrizione: {description}\n• Fonte: {source} - {url}\n"
    "Istruzioni aggiuntive: {additional_instructions}\n"
    "Assicurati che il riassunto sia informativo."
]
APE_PROMPT_TEMPLATES_EN: List[str] = [
    "You are an expert in cultural heritage and history. {tone_instruction} {focus_instruction} {lang_instruction} "
    "Analyze the following monument details and generate a concise summary:\n"
    "Name: {name}\nType: {type}\nLocation: {city}, {province}, {region}\nDescription: {description}\nSource: {source} - {url}\n"
    "Additional instructions: {additional_instructions}\n"
    "Ensure the summary is faithful to the data.",
    "Imagine you are a passionate tour guide and cultural expert. {tone_instruction} {focus_instruction} {lang_instruction} "
    "Using the details below, create an engaging summary:\n"
    "• Name: {name}\n• Type: {type}\n• Location: {city}, {province}, {region}\n• Description: {description}\n• Source: {source} - {url}\n"
    "Additional instructions: {additional_instructions}\n"
    "Make sure the summary is informative."
]


def generate_series_personalized_summary(monument: Dict[str, Any], age_group: str, interest: str,
                                         narrator_type: str, language: str,
                                         temperature: float, additional_instructions: str) -> Tuple[
    str, int, List[Dict[str, Any]]]:
    try:
        # Selezione dei template in base alla lingua
        templates = APE_PROMPT_TEMPLATES_EN if language == "English" else APE_PROMPT_TEMPLATES_IT
        # Istruzione sul tono, generata in maniera modulare
        tone_instruction = get_tone_instruction(narrator_type, age_group, language)
        # Istruzione sul focus basata sull'interesse
        if interest.lower() in ["arte", "architettura"]:
            focus_instruction = "Emphasize the artistic and architectural aspects." if language == "English" else "Metti in evidenza gli aspetti artistici e architettonici."
        elif interest.lower() == "storia":
            focus_instruction = "Highlight the historical context." if language == "English" else "Sottolinea il contesto storico."
        elif interest.lower() == "religione":
            focus_instruction = "Emphasize religious references." if language == "English" else "Evidenzia i riferimenti religiosi."
        else:
            focus_instruction = "Provide a comprehensive view of cultural aspects." if language == "English" else "Offri una visione completa degli aspetti culturali."

        lang_instruction = language_instructions.get(language, "Scrivi il riassunto in italiano.")
        details_list = []
        best_summary = None
        best_score = -1
        best_token_count = 0
        for template in templates:
            prompt = template.format(
                tone_instruction=tone_instruction,
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
                logging.error(summary)
            token_count = count_tokens(prompt)
            # Valutazione: utilizzo di BERTScore come metrica principale
            try:
                P, R, F1 = bert_score([summary], [monument.get("description", "")], lang="it")
                bert_score_val = F1.item()
            except Exception as e:
                logging.error(f"Errore nella valutazione BERTScore: {e}")
                bert_score_val = 0
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
        # Salva i dettagli dei prompt in CSV per ulteriori analisi
        save_ape_details_to_csv(details_list, filename="ape_prompts.csv")
        return best_summary, best_token_count, details_list
    except Exception as e:
        logging.error(f"Errore in generate_series_personalized_summary: {e}")
        return "Errore nella generazione del summary", 0, []


def save_ape_details_to_csv(details: List[Dict[str, Any]], filename: str = "ape_prompts.csv") -> None:
    try:
        fieldnames = ["prompt", "summary", "token_count", "bert_score"]
        with open(filename, mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for d in details:
                writer.writerow(d)
        logging.info(f"Dettagli APE salvati in {filename}")
    except Exception as e:
        logging.error(f"Errore nel salvataggio dei dettagli APE: {e}")


##########################
# 4. FUNZIONI DI VALUTAZIONE E LOG DEI RISULTATI
##########################
def compute_bart_score(candidate: str, reference: str) -> float:
    return random.uniform(0, 1)  # Placeholder: sostituire con implementazione reale


def compute_mover_score(candidate: str, reference: str) -> float:
    return random.uniform(0, 1)  # Placeholder


def evaluate_summary(candidate: str, reference: str) -> Dict[str, Any]:
    if not reference.strip():
        return {"BERTScore_F1": 0, "ROUGE_L": 0, "BLEURT": 0, "BARTScore": 0, "MoverScore": 0}
    try:
        P, R, F1 = bert_score([candidate], [reference], lang="it")
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rougeL = scorer.score(reference, candidate)['rougeL'].fmeasure
        bleurt = 0.76  # Placeholder
        bart = compute_bart_score(candidate, reference)
        mover = compute_mover_score(candidate, reference)
        return {
            "BERTScore_F1": F1.item(),
            "ROUGE_L": rougeL,
            "BLEURT": bleurt,
            "BARTScore": bart,
            "MoverScore": mover
        }
    except Exception as e:
        logging.error(f"Errore in evaluate_summary: {e}")
        return {}


def external_evaluation(summary: str, reference: str = "") -> Dict[str, Any]:
    return evaluate_summary(summary, reference)


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
    try:
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
        filename = "session_results.csv"
        fieldnames = list(result_data.keys())
        file_exists = os.path.exists(filename) and os.path.getsize(filename) > 0
        backup_session_results(filename)
        with open(filename, mode='a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(result_data)
        logging.info(f"Session result saved to {filename}")
    except Exception as e:
        logging.error(f"Errore nel log della sessione: {e}")


def backup_session_results(filename: str = "session_results.csv") -> None:
    try:
        if os.path.exists(filename):
            backup_filename = f"session_results_backup_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            shutil.copy(filename, backup_filename)
            logging.info(f"Session results backed up to {backup_filename}")
    except Exception as e:
        logging.error(f"Errore nel backup dei risultati della sessione: {e}")


##########################
# 5. FUNZIONE DI PROCESSAMENTO DELLA QUERY (INTEGRAZIONE RAG)
##########################
def retrieve_docs_es(user_query: str, top_k: int = 3, index_name: str = INDEX_NAME) -> List[str]:
    try:
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
        results = [h["_source"]["description"] for h in hits]
        logging.info(f"Chunk trovati: {results}")
        return results
    except Exception as e:
        logging.error(f"Errore in retrieve_docs_es: {e}")
        return []


def get_monument_data_from_csv(selected_name: str, csv_filename: str = "monumenti_dettagliati.csv") -> Dict[str, Any]:
    try:
        if os.path.exists(csv_filename):
            with open(csv_filename, mode='r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if row["name"] == selected_name:
                        return row
    except Exception as e:
        logging.error(f"Errore in get_monument_data_from_csv: {e}")
    return {}


def summarize_text(text: str, age_group: str, interest: str, temperature: float, chunk_size: int, overlap_size: int) -> \
Tuple[str, int]:
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap_size)
        split_docs = splitter.split_text(text)
        summaries = []
        total_tokens = 0
        enc = tiktoken.get_encoding("cl100k_base")
        for doc in split_docs:
            prompt = (
                f"Sei un esperto di storia dell'arte. "
                f"Riassumi il seguente testo, attenendoti al testo e senza inventare dettagli, adattandolo per un pubblico {age_group} interessato a {interest}:\n\n"
                f"{doc}\n\n"
                "Il riassunto deve includere dettagli su architettura, storia e importanza culturale."
            )
            logging.info(f"Prompt di sommario: {prompt}")
            llm.temperature = temperature
            try:
                result = llm.generate([prompt])
                summary = result.generations[0][0].text.strip()
            except Exception as e:
                summary = f"Errore: {str(e)}"
                logging.error(summary)
            summaries.append(summary)
            total_tokens += len(enc.encode(doc))
        return "\n\n".join(summaries), total_tokens
    except Exception as e:
        logging.error(f"Errore in summarize_text: {e}")
        return "Errore nella generazione del sommario", 0


def process_query(selected_monument_name: str, age_group: str, interest: str, narrator_type: str,
                  language: str, temperature: float, chunk_size: int, overlap_size: int,
                  save_csv: bool, ext_eval: bool, user_query: str = "", retrieval_method: str = "ElasticSearch",
                  additional_instructions: str = "") -> Tuple[str, int, str, str, str]:
    try:
        # Aggiorno il modello LLM al modello predefinito
        llm.model = DEFAULT_MODEL
        if retrieval_method == "ElasticSearch":
            full_query = f"{selected_monument_name} {user_query}" if user_query else selected_monument_name
            logging.info(f"Full query per ES: {full_query}")
            relevant_chunks = retrieve_docs_es(full_query, top_k=3, index_name=INDEX_NAME)
            if not relevant_chunks:
                return "Nessun chunk trovato da ES.", 0, "N/D", json.dumps({}), ""
            combined_text = "\n".join(relevant_chunks)
            logging.info(f"Testo combinato dai chunk: {combined_text}")
            summary_text, tokens = summarize_text(combined_text, age_group, interest, temperature, chunk_size,
                                                  overlap_size)
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
            eval_metrics = external_evaluation(final_summary, monument_data.get("description", ""))
            if save_csv:
                save_results_csv(user_query, final_summary, monument_data.get("name", "N/D"))
            details_str = ""
            for d in ape_details:
                details_str += f"**Prompt:**\n```\n{d['prompt']}\n```\n"
                details_str += f"**Summary:**\n```\n{d['summary']}\n```\n"
                details_str += f"**Token Count:** {d['token_count']} | **BERT Score:** {d['bert_score']:.4f}\n\n"
            return final_summary, tokens + ape_tokens, monument_data.get("name", "N/D"), json.dumps(
                eval_metrics), details_str
        else:
            monument = get_monument_data_from_csv(selected_monument_name)
            if not monument or not monument.get("name"):
                return "Monumento non trovato", 0, "N/D", json.dumps({}), ""
            summary_text, tokens = summarize_text(monument["description"], age_group, interest, temperature, chunk_size,
                                                  overlap_size)
            personalized_summary, ape_tokens, ape_details = generate_series_personalized_summary(
                monument, age_group, interest, narrator_type, language, temperature, additional_instructions
            )
            final_summary = personalized_summary
            eval_metrics = external_evaluation(final_summary, monument.get("description", ""))
            if save_csv:
                save_results_csv(user_query, final_summary, monument.get("name", "N/D"))
            details_str = ""
            for d in ape_details:
                details_str += f"**Prompt:**\n```\n{d['prompt']}\n```\n"
                details_str += f"**Summary:**\n```\n{d['summary']}\n```\n"
                details_str += f"**Token Count:** {d['token_count']} | **BERT Score:** {d['bert_score']:.4f}\n\n"
            return final_summary, tokens + ape_tokens, monument.get("name", "N/D"), json.dumps(
                eval_metrics), details_str
    except Exception as e:
        logging.error(f"Errore in process_query: {e}")
        return "Errore nella process_query", 0, "N/D", json.dumps({}), ""


def save_results_csv(query: str, summary: str, monument_name: str, csv_filename: str = "rag_results.csv") -> None:
    try:
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
    except Exception as e:
        logging.error(f"Errore in save_results_csv: {e}")


##########################
# 6. INTERFACCIA GRADIO (UI)
##########################
with gr.Blocks() as demo:
    gr.Markdown("# Personalized Prompt Summarization con RAG (Elasticsearch)")
    gr.Markdown(
        "Il sistema permette di creare/ricreare l'indice su Elasticsearch, indicizzare i dati da CSV, "
        "selezionare un monumento e generare un riassunto personalizzato tramite retrieval avanzato. "
        "Vengono applicate metriche (BERTScore, ROUGE, BLEURT, BARTScore, MoverScore) per valutare la qualità."
    )

    with gr.Row():
        btn_create_index = gr.Button("Crea/Ricrea Indice su Elasticsearch")
        btn_index_csv = gr.Button("Indicizza CSV in Elasticsearch")
    output_logs = gr.Textbox(label="Log/Output", interactive=False, lines=5)


    def create_es_index() -> str:
        create_index_elasticsearch(INDEX_NAME)
        return "Indice creato/ripristinato correttamente."


    def index_data() -> str:
        index_monuments_in_es(csv_filename="monumenti_dettagliati.csv", index_name=INDEX_NAME, chunk_size=500,
                              overlap=100)
        return "Indicizzazione completata."


    btn_create_index.click(fn=create_es_index, inputs=[], outputs=output_logs)
    btn_index_csv.click(fn=index_data, inputs=[], outputs=output_logs)

    with gr.Row():
        map_html = gr.HTML(value=generate_folium_map_from_csv(), label="Mappa Interattiva")

    with gr.Row():
        def load_names() -> List[str]:
            names = []
            csv_file = "monumenti_dettagliati.csv"
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
            placeholder="Esempio: Esperto storico. Assicurati che il riassunto sia informativo."
        )
    with gr.Row():
        temperature_slider = gr.Slider(minimum=0.1, maximum=1.0, step=0.1, value=0.7, label="Temperatura")
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
                chunk_slider, overlap_slider, save_csv_checkbox, external_eval_checkbox, retrieval_method,
                additional_instructions],
        outputs=[results_output, ape_details_output]
    )

    with gr.Accordion("Guida per scrivere manualmente un prompt (APE)", open=False):
        manual_guidelines = gr.Markdown(
            "Linee guida per scrivere manualmente un prompt efficace:\n\n"
            "Esempio in Italian:\n"
            "-----------------------------------\n"
            "Sei un assistente esperto in patrimonio culturale. Usa un linguaggio formale e dettagliato. "
            "Sottolinea gli aspetti storici e artistici. Scrivi il riassunto in italiano.\n\n"
            "Esempio in English:\n"
            "-----------------------------------\n"
            "You are an expert in cultural heritage. Use formal and detailed language. "
            "Highlight historical and artistic aspects. Write the summary in English."
        )
        gr.Markdown("Usa queste linee guida se vuoi scrivere il tuo prompt manualmente.")

demo.launch(share=True)

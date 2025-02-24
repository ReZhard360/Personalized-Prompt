# Personalized-Prompt
Prompt customization system per user. Summarization for historical monuments
# Prompt Engineering Personalizzato per Interazione Uomo-Macchina

Questo progetto è stato sviluppato nell'ambito della tesi di Laurea Magistrale in Intelligenza Artificiale e della Sicurezza Informatica. L'obiettivo è quello di migliorare l'interazione tra utenti (di vari livelli di competenza) e Grandi Modelli Linguistici (LLM) attraverso un sistema di **Prompt Engineering Personalizzato** integrato con **Retrieval-Augmented Generation (RAG)** e un'analisi automatizzata dei risultati.

## Sommario

Il sistema si compone di:
- **Indicizzazione e Ingestione Dati:** Utilizzo di Elasticsearch per indicizzare dati provenienti da file CSV (es. "monumenti_dettagliati.csv").  
- **Prompt Engineering (APE):** Generazione di prompt personalizzati che tengono conto di variabili come lo stile narrativo, la fascia d'età, la lingua e l'interesse (storia, arte, architettura, religione).  
- **Retrieval-Augmented Generation (RAG):** Recupero dei chunk rilevanti tramite Elasticsearch e invio al modello LLM per la generazione di riassunti.  
- **Valutazione e Logging:** Calcolo di metriche (BERTScore, ROUGE, BLEURT, ecc.) per valutare la qualità dei riassunti e salvataggio dei risultati.
- **Interfaccia Utente:** Un'interfaccia interattiva sviluppata con Gradio per sperimentare il sistema in modo semplice ed intuitivo.
- **Scraping e Arricchimento Dati:** Script per l'estrazione di dati dal sito [eremos.eu](https://eremos.eu/index.php/sicilia/) e arricchimento degli stessi con estratti da Wikipedia.

## Requisiti

- **Python 3.7+**
- **Librerie:**  
  `requests`, `csv`, `time`, `json`, `tiktoken`, `folium`, `gradio`, `logging`, `random`, `shutil`, `asyncio`, `nltk`, `beautifulsoup4`, `elasticsearch`, `langchain`, `langchain_ollama`, `bert_score`, `rouge_score`, `scikit-learn`, `imblearn`  
- **Servizi esterni:**  
  - Elasticsearch (configurabile tramite variabili d'ambiente)  
  - OllamaLLM, che viene lanciato tramite Docker (vedi sotto)  
  - Accesso a Wikipedia e OpenStreetMap

## Installazione e Configurazione

1. **Clonare il repository:**

   ```bash
   git clone https://tuo-repository.git
   cd tuo-repository

import requests
from bs4 import BeautifulSoup
from bs4.element import Comment
from urllib.parse import urlparse, urljoin
import csv
import os
import tiktoken

# Importazioni aggiornate di LangChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import CSVLoader

import gradio as gr

# Inizializza il modello LLaMA
llm = OllamaLLM(
    base_url="http://localhost:11434",
    model="llama3.2",
    temperature=0.7
)

def is_valid_url(url: str) -> bool:
    """Verifica che l'URL sia ben formato."""
    parsed = urlparse(url)
    return all([parsed.scheme, parsed.netloc])

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
        return u" ".join(t.strip() for t in visible_texts)
    except requests.RequestException as e:
        return f"Error occurred while fetching the page: {str(e)}"
    except Exception as e:
        return f"Unexpected error occurred: {str(e)}"

def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

def get_text_splitter(chunk_size: int, overlap_size: int):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap_size
    )

def summarize_text(text, age_group, interest, temperature, chunk_size, overlap_size):
    text_splitter = get_text_splitter(chunk_size=chunk_size, overlap_size=overlap_size)
    split_docs = text_splitter.split_text(text)

    summaries = []
    total_tokens = 0
    enc = tiktoken.get_encoding("cl100k_base")

    for doc in split_docs:
        prompt = (
            f"Sei un esperto di storia dell'arte. "
            f"Riassumi il seguente testo riguardante una chiesa storica, "
            f"adattandolo per un pubblico {age_group} interessato a {interest}:\n\n"
            f"{doc}\n\n"
            "Il riassunto dovrebbe includere dettagli sull'architettura, la storia "
            "e l'importanza culturale della chiesa."
        )
        llm.temperature = temperature
        try:
            # Passa prompt come lista
            result = llm.generate([prompt])
            # Accedi al testo generato
            summary = result.generations[0][0].text
            summaries.append(summary)
            total_tokens += len(enc.encode(doc))
        except Exception as e:
            summaries.append(f"Errore durante la generazione del riassunto: {str(e)}")

    return "\n\n".join(summaries), total_tokens

def get_webpage_summary(name, links_dict, age_group, interest, temperature, chunk_size, overlap_size):
    link = links_dict.get(name)
    if not link:
        return "Link non trovato", 0
    text = extract_text_from_url(link)
    if text.startswith("Error"):
        return text, 0
    summary, total_tokens = summarize_text(text, age_group, interest, temperature, chunk_size, overlap_size)
    return summary, total_tokens

def load_document(file_path: str):
    try:
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith(".txt"):
            loader = TextLoader(file_path)
        elif file_path.endswith(".csv"):
            loader = CSVLoader(file_path)
        else:
            return None, "Unsupported file type"
        docs = loader.load()
        return docs, None
    except Exception as e:
        return None, f"Error occurred: {str(e)}"

def get_document_summary(file_paths: list, age_group: str, interest: str, temperature: float,
                         chunk_size: int, overlap_size: int):
    try:
        summaries = []
        token_counts = []
        for file_path in file_paths:
            docs, error = load_document(file_path)
            if error:
                return error, []
            text_splitter = get_text_splitter(chunk_size=chunk_size, overlap_size=overlap_size)
            split_docs = text_splitter.split_documents(docs)

            for doc in split_docs:
                prompt = (
                    f"Sei un esperto di storia dell'arte. "
                    f"Riassumi il seguente documento riguardante una chiesa storica, "
                    f"adattandolo per un pubblico {age_group} interessato a {interest}:\n\n"
                    f"{doc.page_content}\n\n"
                    "Il riassunto dovrebbe includere dettagli sull'architettura, la storia "
                    "e l'importanza culturale della chiesa."
                )
                llm.temperature = temperature
                try:
                    # Passa prompt come lista
                    result = llm.generate([prompt])
                    # Accedi al testo generato
                    summary = result.generations[0][0].text
                    summaries.append(summary)
                except Exception as e:
                    summaries.append(f"Errore durante la generazione del riassunto: {str(e)}")

            enc = tiktoken.get_encoding("cl100k_base")
            total_tokens = sum(len(enc.encode(doc.page_content)) for doc in split_docs)
            token_counts.append(total_tokens)

        return "\n\n".join(summaries), token_counts
    except Exception as e:
        return f"Error occurred: {str(e)}", []

def ask_about_summary(summary: str, question: str, temperature: float):
    try:
        llm.temperature = temperature
        prompt = (
            f"Basandoti sul seguente riassunto:\n\n{summary}\n\n"
            f"Rispondi alla domanda: {question}"
        )
        # Passa prompt come lista
        result = llm.generate([prompt])
        # Accedi al testo generato
        answer = result.generations[0][0].text
        return answer
    except Exception as e:
        return f"Error occurred: {str(e)}"

def get_cathedral_links():
    url = 'https://it.wikipedia.org/wiki/Categoria:Cattedrali_della_Sicilia'
    headers = {'User-Agent': 'Your Webscraper Bot 1.0'}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')
    links_dict = {}
    for link in soup.select('.mw-category a'):
        href = link.get('href')
        full_url = urljoin('https://it.wikipedia.org', href)
        name = link.text
        links_dict[name] = full_url
    return links_dict

# Interfaccia Gradio aggiornata
with gr.Blocks() as demo:
    gr.Markdown("# Summarizer LLaMA - Cattedrali Siciliane")

    # Inizializza lo stato per i link delle cattedrali
    links_dict_state = gr.State()

    # Pulsante per caricare i link delle cattedrali
    with gr.Row():
        bttn_load_cathedrals = gr.Button("Carica Cattedrali Siciliane", variant='primary')

    # Mostra i nomi delle cattedrali estratte
    with gr.Row():
        link = gr.Dropdown(label='Seleziona una cattedrale', choices=[])

    def update_links(dummy_input=None):
        links_dict = get_cathedral_links()
        if not links_dict:
            return gr.update(choices=[], value=None), "Nessun link trovato", {}
        names = list(links_dict.keys())
        return gr.update(choices=names, value=names[0]), f"{len(names)} cattedrali trovate", links_dict

    bttn_load_cathedrals.click(
        fn=update_links,
        inputs=None,
        outputs=[link, gr.Textbox(label='Stato Caricamento'), links_dict_state]
    )

    # Selezione delle preferenze dell'utente
    with gr.Row():
        with gr.Column(scale=1):
            age_group = gr.Dropdown(
                label='Seleziona la tua fascia d\'età',
                choices=['giovane', 'adulto', 'anziano'],
                value='adulto'
            )
            interest = gr.Dropdown(
                label='Seleziona il tuo interesse',
                choices=['storia', 'arte', 'architettura', 'religione'],
                value='storia'
            )
            temperature = gr.Slider(
                minimum=0.1, maximum=1.0, step=0.1, value=0.7, label='Temperatura'
            )
            chunk = gr.Slider(
                minimum=100, maximum=2000, step=100, value=500, label='Dimensione Chunk'
            )
            overlap = gr.Slider(
                minimum=0, maximum=500, step=50, value=100, label='Overlap'
            )
            bttn_webpage_summ_get = gr.Button("Genera Riassunto", variant='primary')

    with gr.Row():
        with gr.Column():
            webpage_sum = gr.Textbox(label="Riassunto della Cattedrale", show_copy_button=True)
            webpage_token_counts = gr.Textbox(label="Conteggio Token", show_copy_button=True)

    bttn_webpage_summ_get.click(
        fn=get_webpage_summary,
        inputs=[link, links_dict_state, age_group, interest, temperature, chunk, overlap],
        outputs=[webpage_sum, webpage_token_counts]
    )

    # Sezione per domande sul riassunto
    with gr.Row():
        with gr.Column():
            webpage_question = gr.Textbox(label="Fai una domanda sul riassunto")
            bttn_webpage_ask = gr.Button("Chiedi", variant='primary')
            webpage_answer = gr.Textbox(label="Risposta", show_copy_button=True)

    bttn_webpage_ask.click(
        fn=ask_about_summary,
        inputs=[webpage_sum, webpage_question, temperature],
        outputs=webpage_answer
    )

    # Sezione per caricare documenti e generarne il riassunto
    with gr.Row(equal_height=True):
        with gr.Column(scale=4):
            file_paths = gr.File(
                label='Carica Documenti (TXT, CSV, PDF)', file_count='multiple', type='filepath'
            )
        with gr.Column(scale=1):
            bttn_doc_summ_get = gr.Button("Genera Riassunto Documenti", variant='primary')

    with gr.Row():
        with gr.Column():
            doc_sum = gr.Textbox(label="Riassunto del Documento", show_copy_button=True)
            token_counts = gr.Textbox(label="Conteggio Token", show_copy_button=True)

    bttn_doc_summ_get.click(
        fn=get_document_summary,
        inputs=[file_paths, age_group, interest, temperature, chunk, overlap],
        outputs=[doc_sum, token_counts]
    )

    # Sezione per domande sul riassunto del documento
    with gr.Row():
        with gr.Column():
            question = gr.Textbox(label="Fai una domanda sul riassunto del documento")
            bttn_ask = gr.Button("Chiedi", variant='primary')
            answer = gr.Textbox(label="Risposta", show_copy_button=True)

    bttn_ask.click(
        fn=ask_about_summary,
        inputs=[doc_sum, question, temperature],
        outputs=answer
    )

    bttn_clear = gr.Button("Pulisci", variant='secondary')
    bttn_clear.click(
        fn=lambda: [gr.update(value='') for _ in range(15)],
        inputs=None,
        outputs=[
            link, age_group, interest, temperature, chunk, overlap, webpage_sum,
            file_paths, doc_sum, token_counts, question, answer, webpage_question, webpage_answer
        ]
    )

if __name__ == "__main__":
    demo.launch(share=True)

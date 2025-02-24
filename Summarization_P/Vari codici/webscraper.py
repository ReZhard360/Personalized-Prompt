import requests
from bs4 import BeautifulSoup
import re
import csv
from urllib.parse import urljoin, urlparse
import os
import time

class Webscraper:
    def __init__(self, base_url):
        self.base_url = base_url
        self.visited_urls = set()
        self.church_links = []
        self.output_dir = '../output'

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def is_valid_url(self, url):
        """Verifica se l'URL è valido e appartiene al dominio di base."""
        parsed = urlparse(url)
        return bool(parsed.netloc) and bool(parsed.scheme) and self.base_url in url

    def fetch_webpage(self, url):
        try:
            headers = {'User-Agent': 'Your Webscraper Bot 1.0'}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            print(f"Errore durante il fetching di {url}: {e}")
            return None

    def parse_webpage(self, content):
        return BeautifulSoup(content, 'html.parser')

    def scrape_site(self):
        """Inizia il crawling dal base_url."""
        self.crawl(self.base_url)

        # Salva i link delle chiese in un file CSV
        self.save_links_to_csv(self.church_links)

    def crawl(self, url):
        """Crawling ricorsivo delle pagine per trovare link alle chiese."""
        if url in self.visited_urls:
            return
        self.visited_urls.add(url)

        response = self.fetch_webpage(url)
        if response is None:
            return

        soup = self.parse_webpage(response.content)
        time.sleep(1)  # Rispetta i termini di servizio, evitando troppe richieste ravvicinate

        # Estrarre e processare i link sulla pagina
        for link_tag in soup.find_all('a', href=True):
            href = link_tag['href']
            full_url = urljoin(self.base_url, href)
            if not self.is_valid_url(full_url):
                continue
            if full_url in self.visited_urls:
                continue

            # Verifica se il link è relativo a una chiesa (puoi personalizzare il filtro)
            if self.is_church_link(full_url):
                self.church_links.append(full_url)
            else:
                # Continua il crawling su questo link
                self.crawl(full_url)

    def is_church_link(self, url):
        """Determina se un URL è relativo a una chiesa."""
        # Esempio di filtro: verifica se la parola 'chiesa' o 'duomo' è nell'URL
        return 'chiesa' in url.lower() or 'duomo' in url.lower()

    def save_links_to_csv(self, links):
        with open(f'{self.output_dir}/church_links.csv', 'w', encoding='utf-8', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Link'])
            for link in links:
                writer.writerow([link])

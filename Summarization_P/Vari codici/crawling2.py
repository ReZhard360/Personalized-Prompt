import requests
from bs4 import BeautifulSoup
import re
import json
import csv
from urllib.parse import urljoin
import os

class Webscraper:
    def __init__(self, url):
        self.url = url
        self.results = []
        self.output_dir = '../output'
        self.links_csv = f'{self.output_dir}/links.csv'

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def fetch_webpage(self, url):
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response

    def parse_webpage(self, content):
        return BeautifulSoup(content, 'html.parser')

    def scrape_page(self, url):
        try:
            data = self.fetch_webpage(url)
            soup = self.parse_webpage(data.content)
            return soup
        except requests.RequestException as e:
            print(f"Error fetching the webpage: {e}")
            return None

    def scrape_site(self):
        try:
            soup = self.scrape_page(self.url)
            if not soup:
                return

            main_links = self.extract_main_links(soup)
            self.save_links_to_csv(main_links)

            for link in main_links:
                print("Current page:", link)
                soup = self.scrape_page(link)
                if not soup:
                    continue

                church_data, main_text = self.extract_church_data(soup, link)
                if church_data:
                    self.results.append(church_data)

                    filename_base = re.sub(r'\W+', '_', link)
                    self.save_to_json(f'{self.output_dir}/{filename_base}.json', [church_data])
                    self.save_to_csv(f'{self.output_dir}/{filename_base}.csv', [church_data])
                    self.save_to_txt(f'{self.output_dir}/{filename_base}.txt', [church_data])
                    self.save_main_text(f'{self.output_dir}/{filename_base}_main_text.txt', main_text)

            self.save_to_json(f'{self.output_dir}/all_scraped_data.json', self.results)
            self.save_to_csv(f'{self.output_dir}/all_scraped_data.csv', self.results)
            self.save_to_txt(f'{self.output_dir}/all_scraped_data.txt', self.results)

        except Exception as e:
            print(f"An error occurred: {e}")

    def extract_main_links(self, soup):
        content = soup.find("div", {"id": "mw-content-text"})
        links = content.find_all("a", href=True) if content else []
        filtered_links = [link['href'] for link in links if re.match(r'^/wiki/', link['href']) and 'redlink' not in link['href']]
        full_links = [urljoin(self.url, link) for link in filtered_links]
        return full_links

    def save_links_to_csv(self, links):
        with open(self.links_csv, 'w', encoding='utf-8', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Link'])
            for link in links:
                writer.writerow([link])

    def extract_church_data(self, soup, base_url):
        church_data = {
            'page_url': base_url,
            'name': '',
            'description': '',
            'location': '',
            'details': []
        }

        title = soup.find("h1", {"id": "firstHeading"})
        if title:
            church_data['name'] = title.text.strip()

        paragraphs = soup.find_all("p")
        if paragraphs:
            church_data['description'] = ' '.join([para.text.strip() for para in paragraphs[:3]])

        infobox = soup.find("table", {"class": "infobox"})
        if infobox:
            for row in infobox.find_all("tr"):
                header = row.find("th")
                if header:
                    header_text = header.text.strip()
                    value = row.find("td")
                    if value:
                        value_text = value.text.strip()
                        if 'location' in header_text.lower():
                            church_data['location'] = value_text
                        else:
                            church_data['details'].append({header_text: value_text})

        content = soup.find("div", {"id": "mw-content-text"})
        main_text = "\n".join([para.get_text(strip=True) for para in content.find_all("p")]) if content else ""

        if church_data['name'] or church_data['description']:
            return church_data, main_text
        return None, None

    def save_to_json(self, filename, data):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def save_to_csv(self, filename, data):
        with open(filename, mode='w', encoding='utf-8', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Page URL', 'Name', 'Description', 'Location', 'Details'])
            for church in data:
                writer.writerow([
                    church['page_url'],
                    church['name'],
                    church['description'],
                    church['location'],
                    json.dumps(church['details'], ensure_ascii=False)
                ])

    def save_to_txt(self, filename, data):
        with open(filename, 'w', encoding='utf-8') as file:
            for church in data:
                file.write(f"Page URL: {church['page_url']}\n")
                file.write(f"Name: {church['name']}\n")
                file.write(f"Description: {church['description']}\n")
                file.write(f"Location: {church['location']}\n")
                file.write(f"Details: {json.dumps(church['details'], ensure_ascii=False)}\n")
                file.write("\n")

    def save_main_text(self, filename, text):
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(text)

# Example usage: pass the URL of the site to analyze below
scraper = Webscraper("https://it.wikipedia.org/wiki/Duomo_di_Milano")
scraper.scrape_site()


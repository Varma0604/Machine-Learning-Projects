import requests
from bs4 import BeautifulSoup
import csv
import os
import argparse
import random
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WebScraper:
    def __init__(self, url):
        self.url = url
        self.data = []
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
            "Mozilla/5.0 (Linux; Android 10; SM-G973F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.77 Mobile Safari/537.36",
            "Mozilla/5.0 (Linux; Android 11; Pixel 5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Mobile Safari/537.36",
        ]

    def fetch_page(self, url):
        """Fetch the HTML content of the page with a random user-agent."""
        headers = {
            'User-Agent': random.choice(self.user_agents)
        }
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Raise an error for bad responses
            logging.info(f"Successfully fetched {url}")
            return response.text
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching {url}: {e}")
            return None

    def parse_html(self, html):
        """Parse the HTML content and extract data."""
        soup = BeautifulSoup(html, 'html.parser')
        items = soup.find_all('div', class_='item')  # Update based on the actual HTML structure
        for item in items:
            title = item.find('h2').text.strip() if item.find('h2') else 'N/A'
            price = item.find('span', class_='price').text.strip() if item.find('span', class_='price') else 'N/A'
            if self.validate_data(title, price):
                self.data.append({'title': title, 'price': price})
            else:
                logging.warning(f"Invalid data found: Title: {title}, Price: {price}")

    def validate_data(self, title, price):
        """Validate the extracted data."""
        return isinstance(title, str) and title != 'N/A' and isinstance(price, str) and price != 'N/A'

    def save_to_csv(self, filename):
        """Save the scraped data to a CSV file."""
        if not self.data:
            logging.info("No data to save.")
            return
        try:
            with open(filename, mode='w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['title', 'price']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for row in self.data:
                    writer.writerow(row)
            logging.info(f"Data successfully saved to {filename}")
        except IOError as e:
            logging.error(f"Error saving data to CSV: {e}")

    def scrape(self):
        """Main scraping method, supports pagination if required."""
        current_page = 1
        while True:
            logging.info(f"Scraping page {current_page}")
            html = self.fetch_page(self.url.format(current_page))  # Assume URL is formatted for pagination
            if html:
                self.parse_html(html)
                # Check for next page logic here
                if not self.has_next_page(html):  # Implement logic to determine if more pages exist
                    break
                current_page += 1
            else:
                break

    def has_next_page(self, html):
        """Check if there's a next page."""
        soup = BeautifulSoup(html, 'html.parser')
        next_button = soup.find('a', class_='next')  # Update based on the actual HTML structure
        return next_button is not None

def main():
    parser = argparse.ArgumentParser(description="Web Scraper for extracting data from a webpage.")
    parser.add_argument('url', type=str, help='URL of the webpage to scrape (use {} for pagination)')
    parser.add_argument('-o', '--output', type=str, default='output.csv', help='Output CSV file name')
    
    args = parser.parse_args()
    
    scraper = WebScraper(args.url)
    scraper.scrape()
    scraper.save_to_csv(args.output)

if __name__ == "__main__":
    main()

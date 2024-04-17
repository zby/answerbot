
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
import os
from urllib.parse import urlparse

output_dir = 'data/DORA/'

def is_same_domain(url, base_url):
    """ Check if the URL belongs to the same domain as the base URL. """
    domain = urlparse(base_url).netloc
    return urlparse(url).netloc == domain

def download_articles(base_url):
    # Make a directory to store the markdown files
    os.makedirs(output_dir, exist_ok=True)

    # List to store all extracted links
    all_links = {}

    # Fetch the main page
    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the links within the 'projects-wrapper' div
    container = soup.find('div', class_='container-fluid projects-wrapper')
    links = container.find_all('a', href=True)

    # Process each link
    for link in links:
        url = link['href']
        if not url.startswith('http'):
            url = urlparse(base_url)._replace(path=url).geturl()

        # Check if the link is from the same domain
        if is_same_domain(url, base_url):
            # Fetch each linked page
            article_response = requests.get(url)
            article_soup = BeautifulSoup(article_response.text, 'html.parser')

            # Extract the content within the 'projects-wrapper' div
            content_div = article_soup.find('div', class_='container-fluid projects-wrapper')
            if content_div:
                # Convert HTML to Markdown
                markdown_text = md(str(content_div), heading_style="ATX")

                # Save the Markdown text to a file
                file_name = url.split('/')[-1].split('.')[0] + '.md'
                with open(os.path.join(output_dir, file_name), 'w', encoding='utf-8') as file:
                    file.write(markdown_text)
                    print(f'Saved {file_name}')

                # Extract and list all links from this document
                article_links = content_div.find_all('a', href=True)
                for article_link in article_links:
                    all_links[article_link['href']] = 1

    return all_links

# Base URL of the website
base_url = 'https://www.digital-operational-resilience-act.com/DORA_Articles.html'
all_extracted_links = download_articles(base_url)

# Optionally, save the list of all links to a file
with open(output_dir + 'all_links.txt', 'w', encoding='utf-8') as links_file:
    links_file.write('\n'.join(all_extracted_links.keys()))
    print('Saved all_links.txt')

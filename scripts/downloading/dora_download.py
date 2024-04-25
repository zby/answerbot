import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
import html2text
import os
from urllib.parse import urlparse


def is_same_domain(url, base_url):
    """ Check if the URL belongs to the same domain as the base URL. """
    domain = urlparse(base_url).netloc
    return urlparse(url).netloc == domain

def extract_content(html, preamble):
    article_soup = BeautifulSoup(html, 'html.parser')

    # Extract the content within the 'projects-wrapper' div
    content_div = article_soup.find('div', class_='container-fluid projects-wrapper')
    if content_div:
        # Remove the first <h1> tag if present
        first_h1 = content_div.find('h1')
        if first_h1:
            first_h1.decompose()

        # Remove all content after the last <hr> tag
        if preamble:
            index = -2
        else:
            index = -1
        last_hr = content_div.find_all('hr')[index]
        if last_hr:
            for sibling in last_hr.find_all_next():
                sibling.decompose()
            last_hr.decompose()
    return content_div


def download_articles(base_url, output_dir):
    # Make a directory to store the markdown files
    os.makedirs(output_dir, exist_ok=True)

    # List to store all extracted links
    all_links = {}

    # Fetch the main page
    response = requests.get(base_url)
    response.encoding = 'utf-8'  # Explicitly set encoding to UTF-8
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
            article_response.encoding = 'utf-8'  # Again set encoding to UTF-8
            if 'Preamble' in url:
                preamble = True
            else:
                preamble = False
            content_div = extract_content(article_response.text, preamble)
            if content_div:
                converter = html2text.HTML2Text()
                # Avoid breaking links into newlines
                converter.body_width = 0
                # converter.protect_links = True # this does not seem to work
                markdown = converter.handle(str(content_div))
                markdown_text = markdown.strip()

                #markdown_text = md(str(content_div), heading_style="ATX")
                #markdown_text = markdown_text.strip()

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

if __name__ == "__main__":
    # Base URL of the website
    base_url = 'https://www.digital-operational-resilience-act.com/DORA_Articles.html'
    output_dir = 'data/DORA/'
    all_extracted_links = download_articles(base_url, output_dir)

    with open(output_dir + 'all_links.txt', 'w', encoding='utf-8') as links_file:
        links_file.write('\n'.join(all_extracted_links.keys()))
        print('Saved all_links.txt')

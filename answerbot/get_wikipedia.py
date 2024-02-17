import re
import requests
import html2text
import os
import traceback

from bs4 import BeautifulSoup, NavigableString

from answerbot.document import Document, MarkdownDocument

MAX_RETRIES = 3
# BASE_URL = 'https://pl.wikipedia.org/wiki/'
# API_URL = 'https://pl.wikipedia.org/w/api.php'
BASE_URL = 'https://en.wikipedia.org/wiki/'
API_URL = 'https://en.wikipedia.org/w/api.php'

class WikipediaDocument(Document):
    def extract_text(self):
        return self.content

    def section_titles(self):
        headings = re.findall(r'== (.*?) ==', self.content)
        return headings

    def lookup(self, keyword):
        text = self.content
        keyword_escaped = re.escape(keyword)
        match = re.search(keyword_escaped, text, re.IGNORECASE)
        if match is None:  # Keyword not found
            return None
        index = match.span()[0]

        # Determine the start and end points for the extraction
        start = max(index - self.chunk_size // 1, 0)
        end = min(index + len(keyword) + self.chunk_size // 1, len(text))
        surrounding_text = text[start:end]

        # Adjust start point to make sure it doesn't extend past a section boundary
        prev_section_boundary = text.rfind('\n==', start, index)
        if prev_section_boundary != -1:  # Found a previous section boundary
            start = prev_section_boundary + 1

        # Trim the surrounding text after the last '.' character after the keyword
        last_dot_index = text.rfind('.', index, end)
        if last_dot_index != -1:
            end = last_dot_index + 1
        # Trim the surrounding before after the first '.' character before the keyword
        first_dot_index = text.find('.', start, index)
        if first_dot_index != -1:
            start = first_dot_index + 1

        return text[start:end].strip()


class ContentRecord:
    def __init__(self, document, retrieval_history):
        self.document = document
        self.retrieval_history = retrieval_history
    @classmethod
    def load_from_disk(self, title, chunk_size):
        """
        Load a ContentRecord from saved wikitext and retrieval history files based on a given title.

        Returns:
        - ContentRecord: A ContentRecord object reconstructed from the saved files.
        """
        directory = "data/wikipedia_pages"
        sanitized_title = title.replace("/", "_").replace("\\", "_")  # To ensure safe filenames
        sanitized_title = sanitized_title.replace(" ", "_")
        wikitext_filename = os.path.join(directory, f"{sanitized_title}.md")
        history_filename = os.path.join(directory, f"{sanitized_title}.retrieval_history")

        # Load wikitext content
        with open(wikitext_filename, "r", encoding="utf-8") as f:
            document_content = f.read()

        # Load retrieval history
        retrieval_history = []
        with open(history_filename, "r", encoding="utf-8") as f:
            for line in f:
                retrieval_history.append(line.strip())

        document = MarkdownDocument(
            document_content, chunk_size=chunk_size)
        return ContentRecord(document, retrieval_history)

class WikipediaApi:
    def __init__(self, max_retries=MAX_RETRIES, chunk_size=1024, base_url=BASE_URL, api_url=API_URL):
        self.max_retries = max_retries
        self.chunk_size = chunk_size
        self.base_url = base_url
        self.api_url = api_url

    def bracket_links_in_content(self, content, links):
        # Sort links by length in descending order to replace longer links first
        sorted_links = sorted(links, key=len, reverse=True)
        for link in sorted_links:
            # Escape special characters in link for regex pattern
            escaped_link = re.escape(link)
            # Replace occurrences of the link in the content with link surrounded by brackets.
            # Ensure that our match is not already surrounded by brackets.
            content = re.sub(rf'(?<!\[)\b{escaped_link}\b(?!\])', f'[[{link}]]', content)
        return content

    @classmethod
    def clean_html_and_textify(self, html):

        # remove table of content
        soup = BeautifulSoup(html, 'html.parser')
        content = soup.find('div', id='bodyContent')

        for tag in content.find_all():
            if tag.name == 'a' and 'action=edit' in tag.get('href', ''):
                tag.decompose()
            if tag.name == 'div' and tag.get('id') == 'toc':
                tag.decompose()
            if tag.name == 'div' and 'vector-body-before-content' in tag.get('class', []):
                tag.decompose()
            if tag.name == 'div' and 'infobox-caption' in tag.get('class', []):
                tag.decompose()
            if tag.name == 'img':
                tag.decompose()
            if tag.name == 'span' and 'hide-when-compact' in tag.get('class', []):
                tag.decompose()
            if tag.name == 'span' and 'mw-editsection' in tag.get('class', []):
                tag.decompose()
            if tag.name == 'div' and tag.get('id') == 'mw-fr-revisiondetails-wrapper':
                tag.decompose()
            if tag.name == 'figure':
                tag.decompose()

        # Remove some metadata - we need compact information
        search_text = "This article relies excessively on"
        text_before_anchor = "relies excessively on"
        required_href = "/wiki/Wikipedia:Verifiability"

        # Find all <div> elements with class 'mbox-text-span'
        for div in content.find_all('div', class_='mbox-text-span'):
            gtex = div.get_text().strip()
            if search_text in div.get_text():
                for a_tag in div.find_all('a', href=required_href):
                    preceding_text = ''.join([str(sibling) for sibling in a_tag.previous_siblings if isinstance(sibling, NavigableString)])
                    if text_before_anchor in preceding_text:
                        div.decompose()
                        break  # Stop checking this div, as we found a match
        for div in content.find_all('div', class_='mbox-text-span'):
            print(div)
        modified_html = str(content)

        converter = html2text.HTML2Text()
        # Avoid breaking links into newlines
        converter.body_width = 0
        # converter.protect_links = True # this does not seem to work
        markdown = converter.handle(modified_html)
        cleaned_content = markdown.strip()
        return cleaned_content

    def get_page(self, title):
        retrieval_history = []
        retries = 0

        while retries < self.max_retries:
            url = self.base_url + title
            response = requests.get(url)
            if response.status_code == 404:
                retrieval_history.append(f"Page '{title}' does not exist.")
                break
            elif response.status_code == 200:
                response.raise_for_status()
                html = response.text
                cleaned_content = self.clean_html_and_textify(html)

                document = MarkdownDocument(cleaned_content, chunk_size=self.chunk_size)
                retrieval_history.append(f"Successfully retrieved '{title}' from Wikipedia.")

                return ContentRecord(document, retrieval_history)
            else:
                retrieval_history.append(f"HTTP error occurred: {response.status_code}")
                break
            retries += 1
        retrieval_history.append(f"Retries exhausted. No options available.")
        return ContentRecord(None, retrieval_history)

    def search(self, search_query):
        params = {
            'action': 'query',
            'format': 'json',
            'list': 'search',
            'srsearch': search_query,
            'srlimit': 10,  # Limit the number of results
        }

        try:
            response = requests.get(self.api_url, params=params)
            response.raise_for_status()
            data = response.json()

            search_results = [item['title'] for item in data['query']['search']]
            squared_results = [f"[[{result}]]" for result in search_results]
            search_history = [f"Wikipedia search results for query: '{search_query}' are: " + ", ".join(squared_results)]

            if search_results:
                content_record = self.get_page(search_results[0])
                combined_history = search_history + content_record.retrieval_history
                return ContentRecord(content_record.document, combined_history)
            else:
                return ContentRecord(None, search_history)

        except requests.exceptions.HTTPError as e:
            search_history = [f"HTTP error occurred during search: {e}"]
            return ContentRecord(None, search_history)
        except Exception as e:
            stack_trace = traceback.format_exc()
            return ContentRecord(None, [stack_trace])



# Example usage
if __name__ == "__main__":
    scraper = WikipediaApi(chunk_size=800)
    title = "Shirley Temple"
    content_record = scraper.search(title)
    if content_record.document:
        print(f"Searching for {title}:\n")
        document = content_record.document
        print(document.first_chunk())
        print('\nLooking up ## Diplomatic career:\n')
        print(document.lookup('## Diplomatic career'))
    else:
        print("No document found")

    print('\n')
    print('------------------\n')
    exit()

    title = "Oxygen"
    content_record = scraper.search(title)
    if content_record.document:
        print(f"Searching for {title}:\n")
        document = content_record.document
        print(document.first_chunk())
        print('\nLooking up atomic weight:\n')
        print(document.lookup('atomic weight'))
        print('\nSection titles:\n')
        print(document.section_titles())
        print("\nRetrieval History:")
        for record in content_record.retrieval_history:
            print(record)
    else:
        print("No document found")

    print('\n')
    print('------------------\n')


#    search_query = "Machine learning"
#    search_record = scraper.search(search_query)
#    if search_record.document:
#        print(search_record.document.first_chunk())
#        print("\nRetrieval History:")
#        for record in search_record.retrieval_history:
#            print(record)
#    else:
#        print("No document found")

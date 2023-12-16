import re
import requests
import html2text
import os

from bs4 import BeautifulSoup, NavigableString

from answerbot.document import Document, MarkdownDocument

MAX_RETRIES = 3
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
    def __init__(self, max_retries=MAX_RETRIES, chunk_size=1024, api_url=API_URL):
        self.max_retries = max_retries
        self.chunk_size = chunk_size
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
        toc_element = soup.find('div', {'id': 'toc'})
        if toc_element:
            toc_element.decompose()
        # Find all <span> tags with the class 'hide-when-compact'
        for span in soup.find_all('span', class_='hide-when-compact'):
            span.decompose()

        # Remove some metadata - we need compact information
        search_text = "This article relies excessively on"
        text_before_anchor = "relies excessively on"
        required_href = "/wiki/Wikipedia:Verifiability"

        # Find all <div> elements with class 'mbox-text-span'
        for div in soup.find_all('div', class_='mbox-text-span'):
            gtex = div.get_text().strip()
            if search_text in div.get_text():
                for a_tag in div.find_all('a', href=required_href):
                    preceding_text = ''.join([str(sibling) for sibling in a_tag.previous_siblings if isinstance(sibling, NavigableString)])
                    if text_before_anchor in preceding_text:
                        div.decompose()
                        break  # Stop checking this div, as we found a match
        for div in soup.find_all('div', class_='mbox-text-span'):
            print(div)
        modified_html = str(soup)

        converter = html2text.HTML2Text()
        # Avoid breaking links into newlines
        converter.body_width = 0
        # converter.protect_links = True # this does not seem to work
        markdown = converter.handle(modified_html)
        # replace the edit links with an empty string
        # __TODO__: move that to MarkdownDocument
        edit_link_pattern = r'\[\[edit\]\(/w/index\.php\?title=.*?\)]'
        cleaned_content = re.sub(edit_link_pattern, '', markdown)
        # Regex pattern to match the image links
        pattern = r'\[!\[.*?\]\((?:.*?[^\\]|.*?)\)\]\((?:.*?[^\\]|.*?)\)'
        # sometimes urls have escaped parenthesis inside them:
        # /wiki/File:Glasto2023_Guns_%27N%27_Roses_\(sans_Dave_Grohl\).jpg
        # !!!
        cleaned_content = re.sub(pattern, '', cleaned_content)
        return cleaned_content

    def get_page(self, title):
        retrieval_history = []
        retries = 0

        while retries < self.max_retries:
            params = {
                'action': 'query',
                'format': 'json',
                'titles': title,
                'prop': 'revisions',
                'rvprop': 'content',
                'rvparse': True,  # Get parsed HTML content
                'redirects': True,
            }

            try:
                response = requests.get(self.api_url, params=params)
                response.raise_for_status()
                data = response.json()
                pages = data['query']['pages']

                if '-1' in pages:  # Page does not exist
                    retrieval_history.append(f"Page '{title}' does not exist.")
                    break

                page = next(iter(pages.values()))

                # Check for disambiguation and redirect (handle as before)

                # Extract HTML content
                if 'revisions' in page and page['revisions']:
                    html = page['revisions'][0]['*']
                    # directory = "data/wikipedia_pages"
                    # sanitized_title = title.replace("/", "_").replace("\\", "_")  # To ensure safe filenames
                    # sanitized_title = sanitized_title.replace(" ", "_")
                    # html_filename = os.path.join(directory, f"{sanitized_title}.html")
                    # with open(html_filename, 'w', encoding='utf-8') as file:
                    #     file.write(html)

                    cleaned_content = self.clean_html_and_textify(html)

                    document = MarkdownDocument(cleaned_content, self.chunk_size)
                    retrieval_history.append(f"Successfully retrieved '{title}' from Wikipedia.")
                else:
                    document = None
                    retrieval_history.append(f"No content found for '{title}'.")
                    break

                return ContentRecord(document, retrieval_history)

            except requests.exceptions.HTTPError as e:
                retrieval_history.append(f"HTTP error occurred: {e}")
                break
            except Exception as e:
                retrieval_history.append(str(e))
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
            search_history = [str(e)]
            return ContentRecord(None, search_history)



# Example usage
if __name__ == "__main__":
    scraper = WikipediaApi(chunk_size=300)
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

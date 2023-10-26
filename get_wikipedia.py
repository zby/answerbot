import wikipedia
import re

from document import Document

MAX_RETRIES = 3
LANGUAGE = 'en'


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

class WikipediaApi:
    def __init__(self, language=LANGUAGE, max_retries=MAX_RETRIES, chunk_size=1024):
        self.language = language
        self.max_retries = max_retries
        self.chunk_size = chunk_size

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

    def get_page(self, title):
        retrieval_history = []
        retries = 0

        while retries < self.max_retries:
            try:
                page = wikipedia.page(title, auto_suggest=False)
                retrieval_history.append(f"Successfully retrieved '{title}' from Wikipedia.")
                if page.content:
                    document = WikipediaDocument(page.content, chunk_size=self.chunk_size)
                    marked_content = self.bracket_links_in_content(page.content, page.links)
                    document = WikipediaDocument(marked_content, chunk_size=self.chunk_size)
                else:
                    document = None
                return ContentRecord(document, retrieval_history)
            except wikipedia.DisambiguationError as e:
                retrieval_history.append(f"Retrieved disambiguation page for '{title}'. Options: {', '.join(e.options)}")
                title = e.options[0]
            except wikipedia.RedirectError as e:
                retrieval_history.append(f"{title} redirects to {e.title}")
                title = e.title
            except wikipedia.PageError:
                retrieval_history.append(f"Page '{title}' does not exist.")
                break
            retries += 1

        retrieval_history.append(f"Retries exhausted. No options available.")
        return ContentRecord(None, retrieval_history)

    def search(self, search_query):
        search_results = wikipedia.search(search_query, results=10)
        squared_results = [f"[[{result}]]" for result in search_results]
        search_history = [f"Wikipedia search results for query: '{search_query}' is: " + ", ".join(squared_results)]

        if search_results:
            content_record = self.get_page(search_results[0])
            combined_history = search_history + content_record.retrieval_history
            return ContentRecord(content_record.document, combined_history)
        else:
            return ContentRecord(None, search_history)


# Example usage
if __name__ == "__main__":
    scraper = WikipediaApi()
    title = "Python"
    content_record = scraper.get_page(title)
    if content_record.document:
        print(f"Document for {title}:\n")
        print(content_record.document.first_chunk())
        print("\nRetrieval History:")
        for record in content_record.retrieval_history:
            print(record)
    else:
        print("No document found")

    search_query = "Machine learning"
    search_record = scraper.search(search_query)
    if search_record.document:
        print(search_record.document.first_chunk())
        print("\nRetrieval History:")
        for record in search_record.retrieval_history:
            print(record)
    else:
        print("No document found")

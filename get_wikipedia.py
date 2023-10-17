import wikipedia
import re

from document import Document

MAX_RETRIES = 3
LANGUAGE = 'en'


class WikipediaDocument(Document):
    def extract_text(self):
        # replace consecutive whitespaces with one space
        return ' '.join(self.content.split())
        #        # Replace any number of consecutive whitespaces that include newline(s) with one newline
        #        text = re.sub(r'\s*[\r\n]+\s*', '\n', self.content)
        #        # Replace any number of consecutive whitespaces (except newlines) with one space
        #        text = re.sub(r'[ \t]+', ' ', text)
        #        # Simplistic wikitext to plain text transformation
        #        text = re.sub(r'\'\'\'(.*?)\'\'\'', r'\0', text)  # Bold to plain
        return text

    def section_titles(self):
        headings = re.findall(r'== (.*?) ==', self.content)
        return headings

    def lookup(self, keyword):
        # Search for the keyword in the text
        text = self.text
        index = text.find(keyword)
        if index == -1:  # Keyword not found
            return None

        # Determine the start and end points for the extraction
        start = max(index - self.chunk_size // 1, 0)
        end = min(index + len(keyword) + self.chunk_size // 1, len(self.text))
        #surrounding_text = text[start:end]

        # Adjust start point to make sure it doesn't extend past a section boundary
        prev_section_boundary = text.rfind('==', start, index)
        if prev_section_boundary != -1:  # Found a previous section boundary
            the_other_boundary = text.rfind('==', start, prev_section_boundary)
            if the_other_boundary != -1:
                start = the_other_boundary

        # Trim the surrounding text after the last '.' character after the keyword
        last_dot_index = text.rfind('.', index, end)
        if last_dot_index != -1:
            end = last_dot_index + 1

        return text[start:end].strip()
class ContentRecord:
    def __init__(self, document, retrieval_history):
        self.document = document
        self.retrieval_history = retrieval_history

class WikipediaApi:
    def __init__(self, language=LANGUAGE, max_retries=MAX_RETRIES):
        self.language = language
        self.max_retries = max_retries

    def get_page(self, title, retrieval_history=None, retries=0):
        if retrieval_history is None:
            retrieval_history = []

        try:
            wikipedia.set_lang(self.language)
            # fixed_title = title.replace(" ", "_")
            page = wikipedia.page(title, auto_suggest=False)
            document = WikipediaDocument(page.content, summary=page.summary)
            retrieval_history.append(f"Successfully retrieved '{title}' from Wikipedia.")
            return ContentRecord(document, retrieval_history)
        except wikipedia.exceptions.DisambiguationError as e:
            options = e.options
            if options:
                retrieval_history.append(f"Title: '{title}' is ambiguous. Possible alternatives: {', '.join(options)}")
                if retries < self.max_retries:
                    first_option = options[0]
                    retrieval_history.append(f"Retrying with '{first_option}' (Retry {retries + 1}/{self.max_retries}).")
                    return self.get_page(first_option, retrieval_history, retries=retries + 1)
                else:
                    retrieval_history.append(f"Retries exhausted. No options available.")
                    return ContentRecord(None, retrieval_history)
            else:
                retrieval_history.append(f"Title: '{title}' is ambiguous but no alternatives found.")
                return ContentRecord(None, retrieval_history)
        except wikipedia.exceptions.HTTPTimeoutError:
            retrieval_history.append("HTTPTimeoutError: Request to Wikipedia timed out.")
            return ContentRecord(None, retrieval_history)
        except Exception as e:
            retrieval_history.append(f"Error: {str(e)}")
            return ContentRecord(None, retrieval_history)

    def search(self, query):
        retrieval_history = []

        try:
            # Perform a search
            wikipedia.set_lang(self.language)
            search_results = wikipedia.search(query)

            if search_results:
                titles = map(lambda x: f"'{x}'", search_results)
                retrieval_history.append(f"Wikipedia search results for query: '{query}' is: " + ", ".join(titles))

                first_result = search_results[0]
                page_record = self.get_page(first_result, retrieval_history)
                return page_record
            else:
                retrieval_history.append(f"No search results found for query: '{query}'")
                return ContentRecord(None, retrieval_history)
        except wikipedia.exceptions.HTTPTimeoutError:
            retrieval_history.append("HTTPTimeoutError: Request to Wikipedia timed out.")
            return ContentRecord(None, retrieval_history)
        except Exception as e:
            retrieval_history.append(f"Error: {str(e)}")
            return ContentRecord(None, retrieval_history)

# Example usage
if __name__ == "__main__":
    wiki_api = WikipediaApi(max_retries=3)

    # Example 1: Retrieve a Wikipedia page and its retrieval history
    title = "Python (programming language)"
    content_record = wiki_api.get_page(title)

    # Access the Wikipedia page and retrieval history
    if content_record.page:
        print("Wikipedia Page Summary:")
        print(content_record.page.summary)
    else:
        print("Page retrieval failed. Retrieval History:")
        for entry in content_record.retrieval_history:
            print(entry)

    # Example 2: Perform a search and retrieve the first search result with retrieval history
    search_query = "Machine learning"
    search_record = wiki_api.search(search_query)

    # Access the Wikipedia page and retrieval history for the first search result
    if search_record.page:
        print("Wikipedia Page Summary:")
        print(search_record.page.summary)
    else:
        print("No Wikipedia page found for the search query. Retrieval History:")
        for entry in search_record.retrieval_history:
            print(entry)

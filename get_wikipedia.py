import wikipedia

MAX_RETRIES = 3
LANGUAGE = 'en'

class ContentRecord:
    def __init__(self, page, retrieval_history):
        self.page = page
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
            fixed_title = title.replace(" ", "_")
            page = wikipedia.page(title, auto_suggest=False)
            retrieval_history.append(f"Successfully retrieved '{title}' from Wikipedia.")
            return ContentRecord(page, retrieval_history)
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

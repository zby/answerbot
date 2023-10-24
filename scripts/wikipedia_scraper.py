import os
import mwclient

# Package constant
DEFAULT_SITE = mwclient.Site('en.wikipedia.org')


class WikipediaScraper:
    def __init__(self, site=DEFAULT_SITE):
        self.site = site
        self.retrieval_history = []

    def get_wikipedia_page(self, title):
        while True:
            page = self.site.pages[title]
            if page.redirect:
                self.retrieval_history.append((title, "redirection"))
                title = page.redirects_to().name
                continue
            if "disambiguation" in page.categories():
                for section in page.sections():
                    for link in section.links():
                        if "(" not in link.name and link.namespace == 0:
                            self.retrieval_history.append((title, "disambiguation"))
                            title = link.name
                            break
                if title != self.retrieval_history[-1][0]:
                    continue
                else:
                    break
            else:
                self.retrieval_history.append((title, "successful retrieval"))
                break
        return page

    def wikipedia_search(self, search_query):
        search_results = self.site.search(search_query, namespace=0, limit=10)  # Adjust limit as needed
        found_titles = [result['title'] for result in search_results]

        # Add found titles to retrieval_history
        for title in found_titles:
            self.retrieval_history.append((title, "search result"))

        # Get the page for the first search result
        if found_titles:
            return self.get_wikipedia_page(found_titles[0])
        else:
            return None



# List of titles
titles = ["Python", "Java", "Einstein"]

# Initialize scraper
scraper = WikipediaScraper()

# Directory to save the pages and histories
directory = "data/wikipedia_pages"
if not os.path.exists(directory):
    os.makedirs(directory)

# Loop through the titles and save the outputs
for title in titles:
    page = scraper.get_wikipedia_page(title)
    wikitext = page.text()
    sanitized_title = title.replace("/", "_").replace("\\", "_")  # To ensure safe filenames
    wikitext_filename = os.path.join(directory, f"{sanitized_title}.wikitext")
    history_filename = os.path.join(directory, f"{sanitized_title}.retrieval_history")

    # Save the wikitext and retrieval history
    with open(wikitext_filename, "w", encoding="utf-8") as f:
        f.write(wikitext)
    with open(history_filename, "w", encoding="utf-8") as f:
        for item in scraper.retrieval_history:
            f.write(f"{item[0]} - {item[1]}\n")

    # Clear the retrieval history for the next title
    scraper.retrieval_history.clear()

print("Data saved successfully!")

import os
import wikipedia

from get_wikipedia import WikipediaDocument, ContentRecord, WikipediaApi

# List of titles
queries = [
    "Python (programming language)",
    "Colorado orogeny",
    "High Plains",
    "High Plains geology",
    "Milhouse Van Houten",
]

# Initialize scraper with an example chunk size (you can adjust this)
scraper = WikipediaApi()  # Adjust chunk_size as needed

# Directory to save the pages and histories
directory = "data/wikipedia_pages"
if not os.path.exists(directory):
    os.makedirs(directory)

# Loop through the titles and save the outputs
for query in queries:
    content_record = scraper.search(query)
    wikitext = content_record.document.content
    sanitized_title = query.replace("/", "_").replace("\\", "_")  # To ensure safe filenames
    sanitized_title = sanitized_title.replace(" ", "_")
    wikitext_filename = os.path.join(directory, f"{sanitized_title}.txt")
    history_filename = os.path.join(directory, f"{sanitized_title}.retrieval_history")

    # Save the wikitext and retrieval history
    with open(wikitext_filename, "w", encoding="utf-8") as f:
        f.write(wikitext)
    with open(history_filename, "w", encoding="utf-8") as f:
        for item in content_record.retrieval_history:  # Updated reference
            f.write(f"{item}\n")  # Adjusted based on the structure of 'retrieval_history'

print("Data saved successfully!")


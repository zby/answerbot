import wikipedia
import os

# List of Wikipedia page names you want to download
PAGES_LIST = [
    "Python (programming language)",
    "Colorado orogeny",
    "High Plains Drifter",
    "High Plains (United States)",
    "Milhouse Van Houten",
]


def download_wikipedia_pages(page_names, save_directory):
    # Ensure the directory exists
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    for page_name in page_names:
        fixed_name = page_name.replace(" ", "_")
        try:
            # Fetch content of the page
            content = wikipedia.page(fixed_name, auto_suggest=False).content

            # Create a filename based on the page name
            filename = os.path.join(save_directory, f"{fixed_name}.txt")

            # Save content to the file
            with open(filename, 'w', encoding='utf-8') as file:
                file.write(content)
            print(f"Downloaded content for '{page_name}' and saved to '{filename}'")

        except wikipedia.exceptions.DisambiguationError as e:
            print(f"DisambiguationError for '{fixed_name}'. There are multiple options: {e.options}")
        except wikipedia.exceptions.PageError:
            print(f"Page '{fixed_name}' does not exist!")
        except Exception as e:
            print(f"An error occurred for '{fixed_name}': {e}")


if __name__ == "__main__":
    # Directory to save the downloaded pages
    target_directory = "data/wikipedia_pages"

    download_wikipedia_pages(PAGES_LIST, target_directory)

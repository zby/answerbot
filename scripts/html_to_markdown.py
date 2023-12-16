import sys
from answerbot.get_wikipedia import WikipediaApi


def process_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            file_content = file.read()

        cleaned_text = WikipediaApi.clean_html_and_textify(file_content)

        print(cleaned_text)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <file_path>")
    else:
        file_path = sys.argv[1]
        process_file(file_path)

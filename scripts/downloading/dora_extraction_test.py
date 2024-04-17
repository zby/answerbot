# Import the extract_content function from the specified module
from scripts.4downloading.dora_download import extract_content


def read_and_process_file(file_path):
    # Read the content of the HTML file
    with open(file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()

    # Process the HTML content using the imported function
    result = extract_content(html_content)

    # Print the result
    print(result)


# Specify the path to the HTML file
file_path = 'data/DORA/Preamble_1_to_10.html'

# Call the function to read and process the file
read_and_process_file(file_path)

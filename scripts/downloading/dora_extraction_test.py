# Import the extract_content function from the specified module
from scripts.downloading.dora_download import extract_content

file_path = 'data/DORA/Preamble_1_to_10.html'

# Read the content of the HTML file
with open(file_path, 'r', encoding='utf-8') as file:
    html_content = file.read()

# Process the HTML content using the imported function
result = extract_content(html_content)

# Print the result
print(result)



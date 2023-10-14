from document import WikipediaDocument

file_path = "tests/data/List_of_wars_involving_Ukraine.wikipedia"

with open(file_path, 'r', encoding='utf-8') as file:
    content = file.read()

document = WikipediaDocument(content)

print(document.first_chunk())


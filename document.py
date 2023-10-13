import re
from bs4 import BeautifulSoup

CHUNK_SIZE = 1024

class Document:
    def __init__(self, html, chunk_size=CHUNK_SIZE):
        self.html = html
        self.chunk_size = chunk_size
        self.soup = BeautifulSoup(html, features="html.parser")
        element = self.soup.find('body')
        self.text = element.get_text() if element else ""

    def first_chunk(self):
        sentences = re.split(r'(?<=[.!?])\s+', self.text)
        sentence = sentences[0]
        chunk = sentence[:self.chunk_size]
        for sentence in sentences:
            # Check if adding another sentence would exceed the CHUNK_SIZE
            if len(chunk) + len(sentence) <= self.chunk_size:
                chunk += sentence + " "
            else:
                break
        return chunk.strip()

    def section_titles(self):
        headings = []
        for tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            headings.extend([element.text for element in self.soup.find_all(tag)])
        return headings

    def lookup(self, keyword):
        first_occurrence_node = None

        def dfs_find_keyword(node):
            nonlocal first_occurrence_node
            if first_occurrence_node:  # If we have already found the keyword, we can skip further processing
                return True

            if keyword in node.text:
                first_occurrence_node = node
                return True

            for child in node.children:
                if isinstance(child, type(node)) and dfs_find_keyword(child):  # Recursively check the children
                    return True

            return False

        dfs_find_keyword(self.soup)

        # If the keyword is not found, return None
        if not first_occurrence_node:
            return None

        # Traverse upwards to find the largest element containing the keyword that fits within CHUNK_SIZE
        current = first_occurrence_node
        while current.parent and len(current.parent.text) <= CHUNK_SIZE:
            current = current.parent

        return current.text

#    def lookup(self, keyword):
#        for sentence in self.sentences:
#            if keyword in sentence:
#                for i in range(0, CHUNK_SIZE):
#                    chunk = sentence[i:i+CHUNK_SIZE]
#                    if keyword in chunk:
#                        return chunk
#        for p in self.soup.find_all("p") + self.soup.find_all("ul"):
#            if keyword in p.text:
#                return p.text


import re
from abc import ABC, abstractmethod

CHUNK_SIZE = 1024


class Document(ABC):
    def __init__(self, content, chunk_size=CHUNK_SIZE, retrival_obs=[], summary=None):
        self.content = content
        self.chunk_size = chunk_size
        self.text = self.extract_text()
        self.summary = summary
        self.retrival_obs = retrival_obs

    @abstractmethod
    def extract_text(self):
        pass

    def first_chunk(self):
        if self.summary:
            text = self.summary
        else:
            text = self.text
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentence = sentences[0]
        chunk = sentence[:self.chunk_size]
        for sentence in sentences[1:]:
            if len(chunk) + len(sentence) <= self.chunk_size:
                chunk += sentence + " "
            else:
                break
        return chunk.strip()

    @abstractmethod
    def section_titles(self):
        pass

    @abstractmethod
    def lookup(self, keyword):
        pass


from bs4 import BeautifulSoup, Tag

class SimpleHtmlDocument(Document):

    def extract_text(self):
        # Extracting text from HTML
        self.soup = BeautifulSoup(self.content, features="html.parser")
        title = self.soup.find('title')
        title_text = title.get_text() if title else ''
        body = self.soup.find('body')
        body_text = ' '.join(body.get_text().split()) if body else ''
        return "\n".join([title_text, body_text])

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

            text = node.text.strip()

            if len(text) < self.chunk_size and keyword in text:
                first_occurrence_node = node
                return True

            if isinstance(node, Tag):
                for child in node.children:
                    if dfs_find_keyword(child):  # Recursively check the children
                        return True

            return False

        dfs_find_keyword(self.soup)

        # If the keyword is not found, return None
        if not first_occurrence_node:
            return None

        # Traverse upwards to find the largest element containing the keyword that fits within CHUNK_SIZE
        current = first_occurrence_node
        while current.parent and len(current.parent.text) <= self.chunk_size:
            current = current.parent

        return current.text

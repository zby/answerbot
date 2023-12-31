import re
from abc import ABC, abstractmethod

CHUNK_SIZE = 1024


class Document(ABC):
    def __init__(self, content, chunk_size=CHUNK_SIZE, retrival_obs=[], summary=None, lookup_word=None, lookup_results=None, lookup_position=0):
        self.content = content
        self.chunk_size = chunk_size
        self.text = self.extract_text()
        self.summary = summary
        self.retrival_obs = retrival_obs
        if lookup_results is None:
            self.lookup_results = []
        else:
            self.lookup_results = lookup_results
        self.lookup_word = lookup_word
        self.lookup_position = lookup_position

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
        chunk = chunk.strip()
        self.position = len(chunk)
        return chunk.strip()

    @abstractmethod
    def section_titles(self):
        pass

    @abstractmethod
    def lookup(self, keyword):
        pass

    def next_lookup(self):
        if not self.lookup_results:
            return None

        # Retrieve the current match
        current_match = self.lookup_results[self.lookup_position]

        # Move to the next position, loop back if at the end
        self.lookup_position = (self.lookup_position + 1) % len(self.lookup_results)

        return current_match

class MarkdownDocument(Document):
    def extract_text(self):
        return self.content

    def section_titles(self):
        headings = re.findall(r'^(###? *.*)', self.content, re.MULTILINE)
        return headings

    def lookup(self, keyword):
        text = self.content
        keyword_escaped = re.escape(keyword)
        matches = list(re.finditer(keyword_escaped, text, re.IGNORECASE))

        if not matches:  # No matches found
            return None

        self.lookup_results = []
        for match in matches:
            index = match.span()[0]

            # Determine the start and end points for the extraction
            start = max(index - self.chunk_size // 2, 0)
            end = min(index + len(keyword) + self.chunk_size // 2, len(text))

            # Adjust start point for section boundary
            prev_section_boundary = text.rfind('\n##', start, index + 3)
            if prev_section_boundary != -1:
                start = prev_section_boundary + 1

            chunk = text[start:end].strip()
            self.lookup_results.append(chunk)

        self.lookup_word = keyword
        self.lookup_position = 0
        return self.next_lookup()


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

            #todo position

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

if __name__ == '__main__':
    CHUNK_SIZE = 300

    print("Example for SimpleHtmlDocument (as of now it does not really work)\n")
    file_path = 'data/wikipedia_pages/Oxygen.html'

    with open(file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()


    doc = SimpleHtmlDocument(html_content, chunk_size=CHUNK_SIZE)

    print(doc.first_chunk())
    print(doc.lookup('atomic weight'))

    print()
    print('=' * 80)
    print("Example for MarkdownDocument\n")

    file_path = 'data/wikipedia_pages/Oxygen.md'

    with open(file_path, 'r', encoding='utf-8') as file:
        md_content = file.read()

    doc = MarkdownDocument(md_content, chunk_size=CHUNK_SIZE)
    print(doc.first_chunk())
    print(doc.lookup('atomic weight'))


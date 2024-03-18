import re
from abc import ABC, abstractmethod

import mistune
from mistune.renderers.markdown import MarkdownRenderer

CHUNK_SIZE = 1024


class Document(ABC):
    def __init__(self, content, chunk_size=CHUNK_SIZE, retrival_obs=[], summary=None, position=0, lookup_word=None, lookup_results=None, lookup_position=0,
                 ref_to_url=None, text_to_url=None):
        self.content = content
        self.chunk_size = chunk_size
        self.text = self.extract_text()
        self.summary = summary
        self.position = position
        if retrival_obs is None:
            retrival_obs = []
        self.retrival_obs = retrival_obs
        if lookup_results is None:
            self.lookup_results = []
        else:
            self.lookup_results = lookup_results
        self.lookup_word = lookup_word
        self.lookup_position = lookup_position
        if ref_to_url is None:
            ref_to_url = {}
        self.ref_to_url= ref_to_url
        if text_to_url is None:
            text_to_url = {}
        self.text_to_url = text_to_url

    @abstractmethod
    def extract_text(self):
        pass

    def read_chunk(self):
        text = self.text
        text = text[self.position:]


        sentences = re.split(r'(?<=[.!?]\s)', text)
        sentence = sentences[0]
        chunk = sentence[:self.chunk_size]
        for sentence in sentences[1:]:
            if len(chunk) + len(sentence) <= self.chunk_size:
                chunk += sentence
            else:
                break
        self.position = self.position + len(chunk)
        a = self.text[self.position:]
        chunk = chunk.strip()
        return chunk

    @abstractmethod
    def section_titles(self):
        pass

    @abstractmethod
    def lookup(self, keyword):
        pass

    def next_lookup(self):
        if not self.lookup_results:
            return None

        current_match = self.lookup_results[self.lookup_position]
        self.position = current_match

        # Move to the next position, loop back if at the end
        self.lookup_position = (self.lookup_position + 1) % len(self.lookup_results)

        return self.read_chunk()

    def resolve_link(self, ref_or_text, chunk_start=None, chunk_end=None):
        if ref_or_text in self.ref_to_url:
            return self.ref_to_url[ref_or_text]
        if ref_or_text in self.text_to_url:
            return self.text_to_url[ref_or_text]
        return None



class MarkdownLinkShortener(MarkdownRenderer):
    def __init__(self):
        super().__init__()
        self.url_to_ref = {}  # Map from URL to reference
        self.ref_to_url = {}  # Map from reference to URL
        self.text_to_url = {}
        self.ref_counter = 1

    def link(self, link, title=None, text=None):
        url = link['attrs']['url']
        # Check if the link is already shortened
        if url not in self.url_to_ref:
            ref = str(self.ref_counter)  # Convert the counter to string to use as reference
            self.url_to_ref[url] = ref
            self.ref_to_url[ref] = url # Add the link to the ref_to_url map
            self.ref_counter += 1
        else:
            ref = self.url_to_ref[url]

        link['attrs']['ref'] = ref

        # Extracting the text of the link from the 'children' attribute
        link_text = ''.join(child['raw'] for child in link['children'] if child['type'] == 'text')
        self.text_to_url[link_text] = url

        return f'[{link_text}]({ref})'


class MarkdownDocument(Document):

    # __init__ runs shorten_urls_in_markdown on content and initializes links dictionaries from its output
    def __init__(self, content, **kwargs):

        renderer = MarkdownLinkShortener()
        markdown = mistune.create_markdown(renderer=renderer)

        kwargs['content'] = markdown(content)
        kwargs['ref_to_url'] = renderer.ref_to_url
        kwargs['text_to_url'] = renderer.text_to_url

        # Use kwargs in call to super().__init__
        super().__init__(**kwargs)

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
            index = match.start()
            start = max(int(index + len(keyword) / 2 - self.chunk_size / 2), 0)

            slice_text = text[start:index + len(keyword)]

            pattern = r'(?:^|\n)(#(?:#*)\s*.+)'
            boundary_match = re.search(pattern, slice_text)
            if boundary_match:
                # there is a section boundary between the possible chunk start and the end of the keyword
                start = start + boundary_match.start(1)
            else:
                pattern = r'\.\s+([^\s])'
                boundary_match = re.search(pattern, slice_text)
                if boundary_match:
                    # there is a sentence boundary between the possible chunk start and the end of the keyword
                    start = start + boundary_match.start(1)

            slice_text = text[start:index]

            self.lookup_results.append(start)

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

    print(doc.read_chunk())
    print(doc.lookup('atomic weight'))

    print()
    print('=' * 80)
    print("Example for MarkdownDocument\n")

    file_path = 'data/wikipedia_pages/Oxygen.md'

    with open(file_path, 'r', encoding='utf-8') as file:
        md_content = file.read()

    doc = MarkdownDocument(md_content, chunk_size=CHUNK_SIZE)
    print(doc.read_chunk())
    print(doc.lookup('atomic weight'))


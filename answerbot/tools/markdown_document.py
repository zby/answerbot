import re

class MarkdownDocument:
    def __init__(self, text='', min_size=100, max_size=512, url_shortener=None):
        self.text = text
        self.position = 0
        self.min_size = min_size
        self.max_size = max_size
        self.url_shortener = url_shortener

        self.lookup_results = []
        self.lookup_word = None
        self.lookup_position = -1


    def set_position(self, position):
        """Set the current reading position."""
        if 0 <= position <= len(self.text):
            self.position = position
        else:
            raise ValueError("Position out of range.")

    def find_good_boundary(self, text, search_from_start=True):
        """Find a good semantic boundary within the given text."""
        # Define boundaries with two different behaviors
        boundaries = [
            (r'\n# ', False),  # Header 1
            (r'\n## ', False),  # Header 2
            (r'\n### ', False),  # Header 3
            (r'\n#### ', False),  # Header 4
            (r'\n##### ', False),  # Header 5
            (r'\n###### ', False),  # Header 6
            (r'\n---', False),  # Horizontal rule
            (r'\n___', False),  # Horizontal rule
            (r'\n\*\*\*', False),  # Horizontal rule
            (r'\n-', False),  # Bullet points
            (r'\n\*', False),  # Bullet points
            (r'\n\d+\.', False),  # Numbered list
            (r'\n>', False),  # Blockquote
            (r'```', False),  # Start or end of a code block
            (r'\n\|', False),  # Start of a table row
            (r'\n\n', False),  # Paragraph ending
            (r'\.\s+', True),  # Sentence ending
            (r'\n', False),  # Newline
            (r'\s+', True),  # White space
        ]

        for pattern, after_match in boundaries:
            matches = re.finditer(pattern, text)

            if not search_from_start:
                matches = reversed(list(matches))

            for match in matches:
                return match.end() if after_match else match.start()

        raise ValueError(f"No good boundary found in:\n{text}\n")

    def read_chunk_with_params(self, start_position, min_size, max_size, search_from_start):
        """Read a chunk of text from the specified start position, with given min and max sizes, stopping at a good semantic boundary."""
        if start_position >= len(self.text):
            return ''  # No more text to read

        # Calculate initial end position
        end_position = min(start_position + max_size, len(self.text))

        # Adjust start position if searching from start
        if search_from_start and start_position > 0:
            boundary_end = min(start_position + max_size - min_size, len(self.text))
            boundary_chunk = self.text[start_position:boundary_end]
            boundary_position = self.find_good_boundary(boundary_chunk, search_from_start=True)
            start_position += boundary_position

        # Adjust end position if not searching from start
        if not search_from_start and end_position < len(self.text):
            boundary_start = start_position + min_size
            boundary_chunk = self.text[boundary_start:end_position]
            boundary_position = self.find_good_boundary(boundary_chunk, search_from_start=False)
            end_position = boundary_start + boundary_position

        # Extract the chunk and update position
        chunk = self.text[start_position:end_position]
        self.set_position(end_position)
        return chunk

    def read_chunk(self):
        """Read a chunk of text from the current position, stopping at a good semantic boundary using predefined class attributes."""
        return self.read_chunk_with_params(self.position, self.min_size, self.max_size, False)

    def lookup(self, keyword):
        """Search for a keyword and store the starting positions of all matches."""
        self.lookup_results = []  # Clear previous matches
        self.lookup_position = -1  # Reset the match index
        self.lookup_word = keyword

        for match in re.finditer(re.escape(keyword), self.text):
            self.lookup_results.append(match.start())  # Save the start position of the match

        # TODO make this more systematic with other variants of Markdown in keywords
        if keyword.startswith('#') and not self.lookup_results:
            # Insert space after the initial sequence of '#' characters using regexp
            spaced_keyword = re.sub(r'^(#+)', r'\1 ', keyword)
            for match in re.finditer(re.escape(spaced_keyword), self.text):
                self.lookup_results.append(match.start())  # Save the start position of the match

        return self.next_lookup()

    def next_lookup(self):
        """Search for the next match and read chunks before and after the match."""
        if not self.lookup_results:
            return None
        self.lookup_position += 1
        if self.lookup_position >= len(self.lookup_results):
            self.lookup_position = -1
        keyword_position = self.lookup_results[self.lookup_position]
        # Define the start position for the chunk
        start_position = max(0, keyword_position - self.max_size // 2)

        # Find a good boundary for the start
        pre_chunk = self.read_chunk_with_params(start_position, self.min_size, keyword_position - start_position, True)
        
        in_chunk = self.text[keyword_position -10:keyword_position+10]

        # Find a good boundary for the end
        post_chunk = self.read_chunk_with_params(keyword_position, self.min_size, self.max_size // 2, False)

        return pre_chunk + post_chunk

    def section_titles(self):
        headings = re.findall(r'^(##? *.*)', self.text, re.MULTILINE)
        return headings


if __name__ == "__main__":
    md = MarkdownDocument(
        "This is a sample markdown text. It contains multiple sentences. ## New Section\nAnother sentence here.",
        min_size=10,
        max_size=50
    )
    chunk = md.read_chunk()
    print(chunk)  # Output should stop at a good boundary
    print()
    chunk = md.read_chunk()
    print(chunk)  # Output should stop at a good boundary
    print()
    chunk = md.read_chunk()
    print(chunk)  # Output should stop at a good boundary
    print()
    print('=' * 80)
    print("Example for MarkdownDocument from file\n")

    file_path = 'data/wikipedia_pages/Oxygen.md'

    with open(file_path, 'r', encoding='utf-8') as file:
        md_content = file.read()

    doc = MarkdownDocument(md_content)
    print(doc.read_chunk())
    print()
    print("Find atomic weight:")
    print(doc.lookup('atomic weight'))


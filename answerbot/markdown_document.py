import re

class MarkdownDocument:
    def __init__(self, text='', min_size=100, max_size=500):
        self.text = text
        self.position = 0
        self.min_size = min_size
        self.max_size = max_size

    def set_position(self, position):
        """Set the current reading position."""
        if 0 <= position < len(self.text):
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
            (r'\.\s', True)  # Period followed by space
        ]

        for pattern, after_match in boundaries:
            matches = re.finditer(pattern, text)

            if not search_from_start:
                matches = reversed(list(matches))

            for match in matches:
                return match.end() if after_match else match.start()

        raise ValueError(f"No good boundary found in:\n{text}\n")

    def read_chunk(self):
        """Read a chunk of text from the current position, stopping at a good semantic boundary."""
        if self.position >= len(self.text):
            return ''  # No more text to read

        end_position = min(self.position + self.max_size, len(self.text))
        chunk = self.text[self.position:end_position]

        chosen_boundary = self.find_good_boundary(chunk, search_from_start=False)

        end_position = self.position + chosen_boundary
        chunk = self.text[self.position:end_position]

        self.position = end_position
        return chunk


    def keyword_search(self, keyword):
        """Search for a keyword and return a chunk of text around it with good boundaries."""
        match = re.search(re.escape(keyword), self.text)
        if not match:
            return None  # Keyword not found

        keyword_position = match.start()

        # Define the start position for the chunk
        start_position = max(0, keyword_position - self.max_size // 2)
        pre_chunk = self.text[start_position:keyword_position]

        # Find a good boundary for the start position
        chosen_start_boundary = self.find_good_boundary(pre_chunk, search_from_start=True)
        start_position = start_position + chosen_start_boundary

        end_position = min(start_position + self.max_size, len(self.text))
        chunk = self.text[start_position:end_position]

        # Find a good boundary for the end position
        chosen_end_boundary = self.find_good_boundary(chunk, search_from_start=False)
        end_position = start_position + chosen_end_boundary
        chunk = self.text[start_position:end_position]
        self.position = end_position

        return chunk


# Example usage
text = "Your markdown text goes here..."
doc = MarkdownDocument(text=text)
chunk = doc.keyword_search("keyword")
print(chunk)

if __name__ == "__main__":
    md = MarkdownDocument(
        "This is a sample markdown text. It contains multiple sentences. ## New Section\nAnother sentence here.",
        min_size=10,
        max_size=50
    )
    chunk = md.read_chunk()
    print(chunk)  # Output should stop at a good boundary
    chunk = md.read_chunk()
    print(chunk)  # Output should stop at a good boundary
    chunk = md.read_chunk()
    print(chunk)  # Output should stop at a good boundary

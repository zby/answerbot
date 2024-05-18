import re

class MarkdownDocument:
    def __init__(self, text='', min_size=100, max_size=500):
        self.text = text
        self.position = 0
        self.min_size = min_size
        self.max_size = max_size

        self.matches = []
        self.current_match_index = -1

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
        end_chunk = self.text[self.position + self.min_size:end_position]

        if end_position < len(self.text):
            chosen_boundary = self.find_good_boundary(end_chunk, search_from_start=False)
            end_position = self.position + self.min_size + chosen_boundary

        chunk = self.text[self.position:end_position]
        self.position = end_position
        return chunk


    def keyword_search(self, keyword):
        """Search for a keyword and return a chunk of text around it with good boundaries."""
        match = re.search(re.escape(keyword), self.text)
        if not match:
            return None  # Keyword not found

        keyword_position = match.start()


    def keyword_search(self, keyword):
        """Search for a keyword and store the starting positions of all matches."""
        self.matches = []  # Clear previous matches
        self.current_match_index = -1  # Reset the match index

        for match in re.finditer(re.escape(keyword), self.text):
            self.matches.append(match.start())  # Save the start position of the match

        return self.next_match()

    def next_match(self):
        if not self.matches:
            return None
        self.current_match_index += 1
        if self.current_match_index > len(self.matches):
            self.current_match_index = -1
        keyword_position = self.matches[self.current_match_index]
        # Define the start position for the chunk
        start_position = max(0, keyword_position - self.max_size // 2)
        pre_chunk = self.text[start_position:keyword_position]

        if start_position > 0:
        # Find a good boundary for the start position
            chosen_start_boundary = self.find_good_boundary(pre_chunk, search_from_start=True)
            start_position = start_position + chosen_start_boundary
        pre_chunk = self.text[start_position:keyword_position]
        self.position = keyword_position
        post_chunk = self.read_chunk()
        return pre_chunk + post_chunk

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
    print('=')

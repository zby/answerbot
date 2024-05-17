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

    def read_chunk(self):
        """Read a chunk of text from the current position, stopping at a good semantic boundary."""
        if self.position >= len(self.text):
            return ''  # No more text to read

        end_position = min(self.position + self.max_size, len(self.text))
        chunk = self.text[self.position:end_position]

        # Define the order of boundaries to search for
        boundaries = [
            r'\n###### ',  # Header 6
            r'\n##### ',   # Header 5
            r'\n#### ',    # Header 4
            r'\n### ',     # Header 3
            r'\n## ',      # Header 2
            r'\n# ',       # Header 1
            r'\.\s'        # Period followed by space
        ]

        # Find the positions of boundaries in the chunk
        boundary_positions = {}
        for pattern in boundaries:
            matches = [m.end() for m in re.finditer(pattern, chunk)]
            if matches:
                boundary_positions[pattern] = matches

        # Find the last good boundary within the limits
        chosen_boundary = None
        for pattern in boundaries:
            if pattern in boundary_positions:
                last_boundary = max((b for b in boundary_positions[pattern] if b >= self.min_size), default=None)
                if last_boundary and last_boundary <= self.max_size:
                    chosen_boundary = last_boundary
                    break
        if chosen_boundary is None:
            raise ValueError("No good boundary found within the specified range.")
        else:
            end_position = self.position + chosen_boundary
            chunk = self.text[self.position:end_position]

        self.position = end_position
        return chunk

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

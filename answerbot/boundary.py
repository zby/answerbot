import re


def find_good_boundary(text, search_from_start=True):
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

    return None

if __name__ == "__main__":
    def evaluate_example(text, search_from_start=True):
        """Evaluate an example by finding the boundary and printing the results."""
        result = find_good_boundary(text, search_from_start)
        direction = "start" if search_from_start else "end"
        if result is not None:
            boundary_text = text[result:]
        else:
            boundary_text = "No boundary found"
        print(f"Text: {text}\nDirection of search: {direction}\nBoundary position: {result} -> {boundary_text}\n")


    examples = [
        ("This is some text.\n# Header 1\nThis is more text.", True),
        ("This is the first sentence. This is the second sentence.", True),
        ("Text before header.\n## Header 2\nText after header.", True),
        ("Sentence one. Sentence two. Sentence three.", False),
        ("No special boundaries here", True),
        ("Start text.\n### Header 3\nMore text. Even more text.", True),
        ("Start text.\n### Header 3\nMore text. Even more text.", False),
        (
        "Introduction text.\n# Main Header\nSection 1 content.\n## Subheader 1\nContent.\n### Subheader 2\nMore content.",
        True),
        ("Initial text.\n#### Header 4\nSubsequent text.", True),
        ("Start of text.\n##### Header 5\nEnd of text.", False)
    ]

    for i, (text, search_from_start) in enumerate(examples, 1):
        print(f"Example {i}:")
        evaluate_example(text, search_from_start)

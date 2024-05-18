import pytest
from answerbot.markdown_document import MarkdownDocument


def test_initialization():
    doc = MarkdownDocument(text="Sample text", min_size=50, max_size=200)
    assert doc.text == "Sample text"
    assert doc.position == 0
    assert doc.min_size == 50
    assert doc.max_size == 200


def test_set_position():
    doc = MarkdownDocument(text="Sample text")
    doc.set_position(5)
    assert doc.position == 5

    with pytest.raises(ValueError):
        doc.set_position(-1)

    with pytest.raises(ValueError):
        doc.set_position(len(doc.text) + 1)


def test_read_chunk_within_bounds():
    text = "# Header 1\nSample text.\n## Header 2\nMore text."
    doc = MarkdownDocument(text=text, min_size=10, max_size=50)
    chunk = doc.read_chunk()
    assert chunk == "# Header 1\nSample text.\n## Header 2"
    assert doc.position == len(chunk)


def test_read_chunk_exceed_bounds():
    text = "# Header 1\nSample text.\n## Header 2\nMore text."
    doc = MarkdownDocument(text=text, min_size=10, max_size=20)
    chunk = doc.read_chunk()
    assert chunk == "# Header 1\nSample text."
    assert doc.position == len(chunk)


def test_read_chunk_no_boundary():
    text = "Sample text with no headers."
    doc = MarkdownDocument(text=text, min_size=5, max_size=20)
    chunk = doc.read_chunk()
    assert chunk == "Sample text with no headers."
    assert doc.position == len(chunk)


def test_keyword_search_found():
    text = "This is a test document.\n## Header 2\nHere is the keyword.\nMore text after keyword."
    doc = MarkdownDocument(text=text, min_size=10, max_size=50)
    chunk = doc.keyword_search("keyword")
    assert "keyword" in chunk
    assert "Header 2" in chunk or "More text after keyword." in chunk


def test_keyword_search_not_found():
    text = "This is a test document with no keyword."
    doc = MarkdownDocument(text=text, min_size=10, max_size=50)
    chunk = doc.keyword_search("keyword")
    assert chunk is None

def test_keyword_search_boundary():
    text = "This is a test document.\n## Header 2\nHere is the keyword.\nMore text after keyword."
    doc = MarkdownDocument(text=text, min_size=10, max_size=50)
    chunk = doc.keyword_search("Header 2")
    assert "Header 2" in chunk
    assert "More text after keyword." in chunk or "This is a test document." in chunk

def test_find_good_boundary_start():
    text = "This is a test document.\n## Header 2\nHere is the keyword.\nMore text after keyword."
    doc = MarkdownDocument(text=text, min_size=10, max_size=50)
    boundary = doc.find_good_boundary(text[:30], search_from_start=True)
    assert boundary is not None

def test_find_good_boundary_end():
    text = "This is a test document.\n## Header 2\nHere is the keyword.\nMore text after keyword."
    doc = MarkdownDocument(text=text, min_size=10, max_size=50)
    boundary = doc.find_good_boundary(text[:50], search_from_start=False)
    assert boundary is not None


def test_find_good_boundary_with_rich_structure():
    text = (
        "# Header 1\n"
        "Some introduction text.\n"
        "## Header 2\n"
        "Details about header 2. More details here.\n"
        "### Header 3\n"
        "Further details about a subsection.\n"
        "More text and more details.\n"
        "## Another Header 2\n"
        "Text under another header 2.\n"
        "#### Header 4\n"
        "Details in a deeper subsection.\n"
        "Even more detailed text here.\n"
        "# Conclusion\n"
        "Final thoughts and summary.\n"
    )
    doc = MarkdownDocument(text=text, min_size=50, max_size=200)

    # Test a chunk that should find "## Header 2" as a good boundary
    chunk = text[:100]
    boundary = doc.find_good_boundary(chunk, search_from_start=False)
    assert boundary is not None
    assert text[boundary-9:boundary] == "\n## Header 2"

    # Test a chunk that should find "### Header 3" as a good boundary
    chunk = text[50:150]
    boundary = doc.find_good_boundary(chunk, search_from_start=False)
    assert boundary is not None
    assert text[50 + boundary-9:50 + boundary] == "\n### Header 3"

    # Test a chunk that should prioritize "## Another Header 2" over a period
    chunk = text[100:300]
    boundary = doc.find_good_boundary(chunk, search_from_start=False)
    assert boundary is not None
    assert text[100 + boundary-18:100 + boundary] == "\n## Another Header 2"

    # Test a chunk that should find "# Conclusion" as a good boundary
    chunk = text[200:]
    boundary = doc.find_good_boundary(chunk, search_from_start=False)
    assert boundary is not None
    assert text[200 + boundary-10:200 + boundary] == "\n# Conclusion"

    # Test a chunk with no good boundary other than a period
    chunk = text[20:40]
    boundary = doc.find_good_boundary(chunk, search_from_start=False)
    assert boundary is not None
    assert text[20 + boundary - 1:20 + boundary + 1] == ". "

# Example usage
if __name__ == "__main__":
    pytest.main()

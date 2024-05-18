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
    print(len(text))
    doc = MarkdownDocument(text=text, min_size=10, max_size=40)
    chunk = doc.read_chunk()
    assert chunk == "# Header 1\nSample text."
    assert doc.position == len(chunk)

@pytest.fixture
def doc():
    sample_text = """# Header 1
Some introductory text.

## Header 2
Details about header 2. 

### Header 3
More details in header 3. 

Another paragraph. Ending this section. 

#### Header 4
Details in header 4. A bit more text.

##### Header 5
Text for header 5. Final notes.

###### Header 6
Last header text. The end."""

    return MarkdownDocument(text=sample_text, min_size=50, max_size=100)

def test_read_until_end(doc):
    chunks = []
    while True:
        chunk = doc.read_chunk()
        if not chunk:
            break
        chunks.append(chunk)
    full_text = ''.join(chunks)
    assert full_text == doc.text

def test_keyword_search(doc):
    keyword = "Header 4"
    result = doc.keyword_search(keyword)
    assert keyword in result
    assert len(result) >= doc.min_size
    assert len(result) <= doc.max_size

def test_keyword_not_found(doc):
    keyword = "Nonexistent keyword"
    result = doc.keyword_search(keyword)
    assert result is None

# Example usage
if __name__ == "__main__":
    pytest.main()

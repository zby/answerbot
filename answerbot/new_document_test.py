import pytest
from markdown_document import MarkdownDocument

def test_read_chunk_boundary():
    text = "This is a sample markdown text. It contains multiple sentences. ## New Section\nAnother sentence here."
    md = MarkdownDocument(text, min_size=10, max_size=50)
    chunk = md.read_chunk()
    assert chunk.endswith("## New Section\n") or chunk.endswith(". It contains multiple sentences. "), "Chunk did not end at expected boundary."

def test_read_chunk_min_size():
    text = "This is a sample markdown text. It contains multiple sentences. ## New Section\nAnother sentence here."
    md = MarkdownDocument(text, min_size=10, max_size=50)
    chunk = md.read_chunk()
    assert len(chunk) >= 10, "Chunk is smaller than the minimum size."

def test_read_chunk_max_size():
    text = "This is a sample markdown text. It contains multiple sentences. ## New Section\nAnother sentence here."
    md = MarkdownDocument(text, min_size=10, max_size=50)
    chunk = md.read_chunk()
    assert len(chunk) <= 50, "Chunk is larger than the maximum size."

def test_read_chunk_no_more_text():
    text = "Short text."
    md = MarkdownDocument(text, min_size=10, max_size=50)
    md.read_chunk()  # Read the whole text
    chunk = md.read_chunk()  # Try to read again
    assert chunk == '', "Expected empty string when no more text to read."

def test_set_position():
    text = "This is a sample markdown text."
    md = MarkdownDocument(text)
    md.set_position(10)
    assert md.position == 10, "Position was not set correctly."

def test_set_position_out_of_range():
    text = "This is a sample markdown text."
    md = MarkdownDocument(text)
    with pytest.raises(ValueError):
        md.set_position(len(text) + 1)

if __name__ == "__main__":
    pytest.main()

from answerbot.document import SimpleHtmlDocument

SMALL_CHUNK_SIZE = 70


def test_simple_html_document_extraction():
    html_content = """
    <html>
        <head><title>Test Page</title></head>
        <body>
            <h1>Main Title</h1>
            <h2>Subtitle</h2>
            <p>This is the first paragraph. It's pretty long and contains a lot of text that should be split
             into chunks. Here's another sentence. And yet another one here. And more and more and more.</p>
            <ul>
                <li>Item 1 with the keyword</li>
                <li>Item 2</li>
            </ul>
            <p>Another paragraph with the keyword.</p>
        </body>
    </html>
    """
    doc = SimpleHtmlDocument(html_content, chunk_size=SMALL_CHUNK_SIZE)
    expected_text = (
        "Test Page\nMain Title Subtitle This is the first paragraph. It's pretty long and contains a lot of "
        "text that should be split into chunks. Here's another sentence. And yet another one here. "
        "And more and more and more. Item 1 with the keyword Item 2 Another paragraph with the keyword.")

    assert doc.text == expected_text
    assert "This is the first paragraph." in doc.first_chunk()
    assert len(doc.first_chunk()) <= SMALL_CHUNK_SIZE
    assert doc.first_chunk().endswith(".") or doc.first_chunk().endswith("!") or doc.first_chunk().endswith("?")

    assert len(doc.lookup("keyword")) <= SMALL_CHUNK_SIZE
    assert "Item 1 with the keyword" in doc.lookup("keyword")
    assert doc.lookup("nonexistent") is None

    assert doc.section_titles() == ['Main Title', 'Subtitle']

if __name__ == "__main__":
    test_simple_html_document_extraction()
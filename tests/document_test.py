from answerbot.document import SimpleHtmlDocument, MarkdownDocument, MarkdownLinkShortener

SMALL_CHUNK_SIZE = 73


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
    assert "This is the first paragraph." in doc.read_chunk()
    assert len(doc.read_chunk()) <= SMALL_CHUNK_SIZE
    assert doc.read_chunk().endswith(".") or doc.read_chunk().endswith("!") or doc.read_chunk().endswith("?")

    assert len(doc.lookup("keyword")) <= SMALL_CHUNK_SIZE
    assert "Item 1 with the keyword" in doc.lookup("keyword")
    assert doc.lookup("nonexistent") is None

    assert doc.section_titles() == ['Main Title', 'Subtitle']


def test_markdown_document_extraction():
    wiki_content = """
## Test Page
### Main Title
#### Subtitle
This is the first paragraph. It's pretty long and contains a lot of text that should be split into chunks. Here's another sentence. And yet another one here. And more and more and more.
* Item 1 with the keyword
* Item 2
Another paragraph with the keyword.
## Section 2
Paragraph with a new_keyword.
"""
    doc = MarkdownDocument(wiki_content, chunk_size=SMALL_CHUNK_SIZE)
    #assert doc.text == wiki_content
    first_chunk = doc.read_chunk()
    assert "This is the first paragraph." in first_chunk
    assert len(doc.read_chunk()) <= SMALL_CHUNK_SIZE
    assert doc.section_titles() == ['## Test Page', '### Main Title', '#### Subtitle', '## Section 2']
    assert "Item 1 with the keyword" in doc.lookup("keyword")
    assert "Paragraph with a new_keyword." in doc.lookup("new_keyword")
    assert doc.lookup("new_keyword").startswith("## Section 2")
    assert doc.lookup("Section 2").startswith("## Section 2")
    assert doc.lookup("## Section 2").startswith("## Section 2")
    assert doc.lookup("nonexistent") is None
    wiki_content = "A (b)"
    doc = MarkdownDocument(wiki_content)
    assert wiki_content in doc.lookup("a")
    assert wiki_content in doc.lookup("A (b)")


def test_lookup():
    content = "This is the first paragraph. This is the second paragraph."
    doc = MarkdownDocument(content, chunk_size=SMALL_CHUNK_SIZE)
    keyword = 'second'
    result = doc.lookup(keyword)
    assert result == "This is the second paragraph."

    content = "This is the first paragraph. \n## This is a section.\n\nThis is the second paragraph."
    doc = MarkdownDocument(content, chunk_size=100)
    keyword = 'second'
    result = doc.lookup(keyword)
    assert result == "## This is a section.\n\nThis is the second paragraph."

    content = "## This is the first section."
    doc = MarkdownDocument(content, chunk_size=100)
    keyword = 'first'
    result = doc.lookup(keyword)
    assert result == "## This is the first section."

    content = "Some text\n## This is the first section."
    doc = MarkdownDocument(content, chunk_size=100)
    keyword = '## This'
    result = doc.lookup(keyword)
    assert result == "## This is the first section."


def test_position():
    content = """123456789
a23456789
b23456789
c23456789
"""
    doc = MarkdownDocument(content, chunk_size=10)
    first_chunk = doc.read_chunk()
    assert first_chunk == "123456789"
    assert doc.read_chunk() == "a23456789"
    assert doc.lookup("b23456789") == "b23456789"
    assert doc.read_chunk() == "c23456789"

    doc.position = 0
    assert doc.lookup("123") == "123456789"


def test_links():
    content = """
A link to [OpenAI](https://www.openai.com), and here is another link to [Google](https://www.google.com).
Don't forget to check [Different text OpenAI](https://www.openai.com) for more information.
"""
    doc = MarkdownDocument(content)
    content = doc.content
    assert("[OpenAI](1)" in content)
    assert("[Google](2)" in content)
    assert("[Different text OpenAI](1)" in content)
    assert(doc.ref_to_url == { '1': 'https://www.openai.com', '2': 'https://www.google.com' })

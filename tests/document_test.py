import unittest

from document import SimpleHtmlDocument

SMALL_CHUNK_SIZE = 70

class TestDocumentPackage(unittest.TestCase):

    def test_simple_html_document_extraction(self):
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
        expected_text = ("Test Page\nMain Title Subtitle This is the first paragraph. It's pretty long and contains a lot of "
                         "text that should be split into chunks. Here's another sentence. And yet another one here. "
                         "And more and more and more. Item 1 with the keyword Item 2 Another paragraph with the keyword.")
        self.assertEqual(doc.text, expected_text)
        self.assertIn("This is the first paragraph.", doc.first_chunk())
        # Making sure the first_chunk doesn't exceed the smaller chunk_size of SMALL_CHUNK_SIZE
        self.assertLessEqual(len(doc.first_chunk()), SMALL_CHUNK_SIZE)
        # Also checking if the chunk ends with a full sentence (i.e., not in between)
        self.assertTrue(doc.first_chunk().endswith(".") or doc.first_chunk().endswith(
            "!") or doc.first_chunk().endswith("?"))

        self.assertLessEqual(len(doc.lookup("keyword")), SMALL_CHUNK_SIZE)
        self.assertIn("Item 1 with the keyword", doc.lookup("keyword"))
        self.assertIsNone(doc.lookup("nonexistent"))

        self.assertEqual(doc.section_titles(), ['Main Title', 'Subtitle'])

if __name__ == "__main__":
    unittest.main()

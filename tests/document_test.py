import unittest
from document import Document, CHUNK_SIZE

class TestDocument(unittest.TestCase):

    def setUp(self):
        self.html_content = """
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
        self.doc = Document(self.html_content, chunk_size=50)

    def test_first_chunk(self):
        self.assertIn("This is the first paragraph.", self.doc.first_chunk())
        # Making sure the first_chunk doesn't exceed the smaller chunk_size of 50
        self.assertLessEqual(len(self.doc.first_chunk()), 50)
        # Also checking if the chunk ends with a full sentence (i.e., not in between)
        self.assertTrue(self.doc.first_chunk().endswith(".") or self.doc.first_chunk().endswith(
            "!") or self.doc.first_chunk().endswith("?"))

    def test_section_titles(self):
        self.assertEqual(self.doc.section_titles(), ["Main Title", "Subtitle"])

    def test_lookup(self):
        self.assertIn("Item 1 with the keyword", self.doc.lookup("keyword"))
        self.assertLessEqual(len(self.doc.lookup("keyword")), CHUNK_SIZE)

if __name__ == '__main__':
    unittest.main()

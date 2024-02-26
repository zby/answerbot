import pytest

from answerbot.wikipedia_tool import WikipediaSearch, ContentRecord
from answerbot.document import MarkdownDocument

class MockWikiApi:
    def get_page(self, title):
        content = f"""# {title}
        
A link to [test](https://www.test.test), and here is another link to [Google](https://www.google.com).
Don't forget to check [Different text OpenAI](https://www.openai.com) for more information.
"""
        doc = MarkdownDocument(content)

        return ContentRecord(doc, [f"Successfully retrieved '{title}'"])


def test_follow_link():
    wiki_api = MockWikiApi()
    wiki_search = WikipediaSearch(wiki_api)

    get_object = WikipediaSearch.Get(title="Some Page", reason="because")

    wiki_search.get(get_object)
    assert isinstance(wiki_search.document, MarkdownDocument)
    assert wiki_search.document.content.startswith("# Some Page")

    follow_object = WikipediaSearch.FollowLink(link='1', reason="because")

    wiki_search.follow_link(follow_object)
    new_content = wiki_search.document.content
    assert new_content.startswith("# https://www.test.test")




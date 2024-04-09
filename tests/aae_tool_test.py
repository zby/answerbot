from unittest.mock import patch
from answerbot.aae_tool import AAESearch, SearchQuery, URL, Lookup, NoArgs

class MockHttpResponse:
    def __init__(self, text, status_code):
        self.text = text
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code != 200:
            raise Exception("HTTP Error")


@patch('answerbot.aae_tool.requests.get')
def test_search(mock_get):
    with open('tests/data/risk_based_approach.html', 'r', encoding='utf-8') as f:
        contents = f.read()
    mock_get.return_value = MockHttpResponse(contents, status_code=200)
    tool = AAESearch()
    result = tool.search_aae(SearchQuery(query='risk-based approach'))
    assert len(result) == 844


@patch('answerbot.aae_tool.requests.get')
def test_search_404(mock_get):
    mock_get.return_value = MockHttpResponse('', status_code=404)
    tool = AAESearch()
    result = tool.search_aae(SearchQuery(query='risk-based approach'))
    assert result == 'page not found'


@patch('answerbot.aae_tool.requests.get')
def test_lookup(mock_get):
    with open('tests/data/recital_71.html', 'r', encoding='utf-8') as f:
        contents = f.read()
    mock_get.return_value = MockHttpResponse(contents, status_code=200)
    tool = AAESearch()
    tool.goto_url(URL(url='https://'))
    result = tool.lookup(Lookup(keyword='Member States'))
    assert 'To ensure a legal framework that promotes innovation' in result


@patch('answerbot.aae_tool.requests.get')
def test_lookup_next(mock_get):
    with open('tests/data/recital_71.html', 'r', encoding='utf-8') as f:
        contents = f.read()
    mock_get.return_value = MockHttpResponse(contents, status_code=200)
    tool = AAESearch()
    tool.goto_url(URL(url='https://'))
    result = tool.lookup(Lookup(keyword='Member States'))
    result = tool.lookup_next(NoArgs())
    assert 'Member States could also fulfill this obligation through participating in already existing regulatory sandboxes' in result



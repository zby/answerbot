from unittest.mock import patch
from answerbot.tools.aae import AAESearch
from answerbot.tools.observation import Observation

class MockHttpResponse:
    def __init__(self, text, status_code):
        self.text = text
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code != 200:
            raise Exception("HTTP Error")


@patch('answerbot.tools.aae.requests.get')
def test_search(mock_get):
    with open('tests/data/risk_based_approach.html', 'r', encoding='utf-8') as f:
        contents = f.read()
    mock_get.return_value = MockHttpResponse(contents, status_code=200)
    tool = AAESearch()
    result = tool.search_aae(query='risk-based approach')
    assert len(result.content) == 844


@patch('answerbot.tools.aae.requests.get')
def test_search_404(mock_get):
    mock_get.return_value = MockHttpResponse('', status_code=404)
    tool = AAESearch()
    result = tool.search_aae(query='risk-based approach')
    assert result.content == 'page not found'


@patch('answerbot.tools.aae.requests.get')
def test_lookup(mock_get):
    with open('tests/data/recital_71.html', 'r', encoding='utf-8') as f:
        contents = f.read()
    mock_get.return_value = MockHttpResponse(contents, status_code=200)
    tool = AAESearch()
    tool.goto_url(url='https://')
    result = tool.lookup(keyword='Member States')
    assert isinstance(result, Observation)
    assert result.operation == 'lookup'
    assert 'To ensure a legal framework that promotes innovation' in result.content


@patch('answerbot.tools.aae.requests.get')
def test_lookup_next(mock_get):
    with open('tests/data/recital_71.html', 'r', encoding='utf-8') as f:
        contents = f.read()
    mock_get.return_value = MockHttpResponse(contents, status_code=200)
    tool = AAESearch()
    tool.goto_url(url='https://')
    result = tool.lookup(keyword='Member States')
    result = tool.lookup_next()
    assert isinstance(result, Observation)
    assert result.operation == 'lookup_next'
    assert 'Member States could also fulfill this obligation through participating in already existing regulatory sandboxes' in result.content
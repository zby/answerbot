from llm_easy_tools import external_function, extraction_model
import requests
import html2text
from bs4 import BeautifulSoup, PageElement, Tag
from answerbot.document import MarkdownDocument

from answerbot.wikipedia_tool import BASE_URL
from answerbot.tools_base import SearchQuery, Lookup, URL, NoArgs, Finish
from tenacity import retry, stop_after_attempt


MAX_RETRIES = 3
BASE_URL = 'https://artificialintelligenceact.eu/'
CHUNK_SIZE = 1024


class AAESearch:
    Finish = Finish

    def __init__(
            self, 
            max_retries=MAX_RETRIES,
            base_url=BASE_URL,
            chunk_size=CHUNK_SIZE
            ):
        self._max_retries = max_retries
        self._base_url = base_url
        self._document = None
        self._chunk_size = chunk_size


    @external_function()
    def search_aae(self, query: SearchQuery):
        """
        Searches Artificial Intelligence Act EU website, and return a list of documents
        matching the query, with short blocks of texts containing the query
        """
        return retry(stop=stop_after_attempt(self._max_retries))(_aae_search)(query.query, self._base_url)

    @external_function()
    def lookup(self, param: Lookup):
        """
        Look up a word on the current page.
        """
        if self._document is None:
            return 'No document defined, cannot lookup'

        text = self._document.lookup(param.keyword)
        if text:
            return (
                    f'Keyword "{param.keyword}" found in current page in' 
                    f'{len(self._document.lookup_results)} places.'
                    f'The first occurence:\n {text}'
                    )
        else:
            return f'Keyword "{param.keyword} not found in current page"'

    @external_function()
    def lookup_next(self, _: NoArgs):
        """
        Jumps to the next occurrence of the word searched previously.
        """
        if self._document is None:
            return 'No document defined, cannot lookup'
        if not self._document.lookup_results:
            return 'No lookup results found'
        text = self._document.next_lookup()
        return (
                f'Keyword "{self._document.lookup_word}" found in: \n'
                f'{text}\n'
                f'{self._document.lookup_position} of {len(self._document.lookup_results)}'
                )

    @external_function()
    def read_chunk(self, _: NoArgs):
        """
        Reads the next chunk of text from the current location in the current document.
        """
        if self._document is None:
            return 'No document defined, cannot read'
        return self._document.read_chunk()

    @external_function()
    def goto_url(self, param: URL):
        """
        Retreive a page at url and saves the retrieved page as the next current page
        """
        response = requests.get(param.url)
        if response.status_code == 404:
            return 'Page does not exist'
        try:
            response.raise_for_status()
        except:
            return 'Could not open the page'
        html = response.text
        cleaned_content = clean_html_and_textify(html)
        document = MarkdownDocument(cleaned_content, chunk_size=self._chunk_size)
        self._document = document
        return f'{param.url} retreived successfully'



def _aae_search(query: str, base_url: str = BASE_URL) -> str:
    params = {
            's': query,
            'et_pb_searchform_submit': 'et_search_proccess',
            'et_pb_include_posts': 'yes',
            'et_pb_include_pages': 'yes',
            }
    response = requests.get(base_url, params=params)
    if response.status_code == 404:
        return 'page not found'
    response.raise_for_status()
    parsed = _aae_search_parse(response.text)
    if len(parsed) == 0:
        # This error message is copied from the page
        # when we make it more generic we'll need to extract the error message from the page dynamically
        return 'No Results Found\n\nThe page you requested could not be found. Try refining your search, or use the navigation above to locate the post.'


    result = '\n\n'.join(f'Title: {title}\nUrl: {url}\n Excerpt: {excerpt}'
                         for url, title, excerpt in parsed)
    return result


def _aae_search_parse(html: str):
    soup = BeautifulSoup(html, features='html.parser')
    post_blocks = soup.find_all('article', class_='et_pb_post')

    result = []

    for block in post_blocks:
        block: PageElement
        title_block = block.find_next('h2', class_ = 'entry-title')
        assert title_block
        anchor = title_block.find_next('a')
        assert isinstance(anchor, Tag)
        url = anchor.get('href')
        title = title_block.text.replace('\n', '')

        post_block = block.find_next('span', class_='excerpt_part')
        assert post_block is not None
        post_text = post_block.text
        
        result.append((url, title, post_text))

    return result
    

def clean_html_and_textify(html):
    # remove table of content
    soup = BeautifulSoup(html, 'html.parser')
    content = soup.find('div', class_='et_pb_post_content')


    converter = html2text.HTML2Text()
    # Avoid breaking links into newlines
    converter.body_width = 0
    # converter.protect_links = True # this does not seem to work
    markdown = converter.handle(content.text)
    cleaned_content = markdown.strip()
    return cleaned_content


if __name__ == "__main__":
    scraper = AAESearch()

    query="training generative AI"
    query="generative AI training steps"

    searchparam = SearchQuery(query=query)
    print(scraper.search_aae(searchparam))
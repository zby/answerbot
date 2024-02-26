from pydantic import BaseModel, Field

from answerbot.get_wikipedia import ContentRecord
from llm_easy_tools import ToolBox, SchemaGenerator, external_function

class WikipediaSearch:
    def __init__(self, wiki_api, cached=False):
        self.wiki_api = wiki_api
        self.cached = cached
        self.document = None

    class Finish(BaseModel):
        """
        Finish the task and return the answer.
        """
        reason: str = Field(description="The reasoning behind the answer")
        answer: str = Field(description="The answer to the user's question")

        def normalized_answer(self):
            answer = self.answer
            if answer.lower() == 'yes' or answer.lower() == 'no':
                answer = answer.lower()
            return answer


    class Search(BaseModel):
        reason: str = Field(description="The reason for searching")
        query: str = Field(description="The query to search for on Wikipedia")

    @external_function()
    def search(self, param: Search):
        """
        Searches Wikipedia, saves the first result page, and informs about the content of that page.
        """
        if not self.cached:
            search_query = param.query
            search_record = self.wiki_api.search(search_query)
        else:
            title = param.query
            chunk_size = self.wiki_api.chunk_size
            search_record = ContentRecord.load_from_disk(title, chunk_size)
        self.document = search_record.document
        return self._retrieval_observations(search_record)

    class Get(BaseModel):
        reason: str = Field(description="The reason for retrieving the page")
        title: str = Field(description="The wikipedia page title")
    @external_function()
    def get(self, param: Get):
        """
        Retrieves a Wikipedia page, saves the result, and informs about the content of that page.
        """

        if self.cached:
            raise Exception("Cached get not implemented")
        search_record = self.wiki_api.get_page(param.title)
        self.document = search_record.document
        return self._retrieval_observations(search_record)

    class Lookup(BaseModel):
        reason: str = Field(description="The reason for searching")
        keyword: str = Field(description="The keyword to search")

    @external_function()
    def lookup(self, param: Lookup):
        """
        Looks up a word on the current page.
        """
        if self.document is None:
            observations = "No document defined, cannot lookup"
        else:
            text = self.document.lookup(param.keyword)
            observations = 'Keyword "' + param.keyword + '" '
            if text:
                num_of_results = len(self.document.lookup_results)
                observations = observations + f"found on current page in {num_of_results} places. The first occurence:\n" + text
            else:
                observations = observations + "not found in current page"
        return observations

    class Next_Lookup(BaseModel):
        reason: str = Field(description="The reason for searching")

    @external_function('next')
    def next_lookup(self, param: Next_Lookup):
        """
        Jumps to the next occurrence of the word searched previously.
        """
        if self.document is None:
            observations = "No document defined, cannot lookup"
        elif not self.document.lookup_results:
            observations = "No lookup results found"
        else:
            text = self.document.next_lookup()
            observations = 'Keyword "' + self.document.lookup_word + '" found in: \n' + text
            num_of_results = len(self.document.lookup_results)
            observations = observations + f"\n{self.document.lookup_position} of {num_of_results} places"
        return observations

    class ReadChunk(BaseModel):
        reason: str = Field(description="The reason for continuing reading in the current place")

    @external_function()
    def read_chunk(self, param: ReadChunk):
        """
        Reads the next chunk of text from the current location in the current document.
        """
        if self.document is None:
            observations = "No document defined, cannot read"
        else:
            observations = self.document.read_chunk()
        return observations

    class FollowLink(BaseModel):
        reason: str = Field(description="The reason for following a link")
        link: str = Field(description="The link to follow")

    @external_function()
    def follow_link(self, param: FollowLink):
        """
        Follows a link from the current page and saves the retrieved page as the next current page
        """
        if self.document is None:
            observations = "No current page, cannot follow "
        else:
            url = self.document.links[param.link]
            search_record = self.wiki_api.get_page(url)
            self.document = search_record.document
            observations = self._retrieval_observations(search_record)
        return observations

    def _retrieval_observations(self, search_record, limit_sections=None):
        observations = ""
        document = search_record.document
        for record in search_record.retrieval_history:
            observations = observations + record + "\n"
        if document is None:
            observations = observations + "No wikipedia page found"
        else:
            sections = document.section_titles()
            if limit_sections is not None:
                sections = sections[:limit_sections]
            sections_list_md = "\n".join(sections)
            observations = observations + f'The retrieved page contains the following sections:\n{sections_list_md}\n'
            observations = observations + "The retrieved page summary starts with:\n" + document.read_chunk() + "\n"
        return observations

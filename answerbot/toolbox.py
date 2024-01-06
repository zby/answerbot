import json
import inspect

from typing import Annotated
from llm_schemas import SchemaGenerator
from pydantic import BaseModel, Field

from answerbot.get_wikipedia import ContentRecord

class ToolResult:
    def __init__(self, tool_name=None, tool_args=None, observations=None, error=None):
        self.tool_name = tool_name
        self.tool_args = tool_args
        self.observations = observations
        self.error = error

class ToolBox:

    def __init__(self, name_mappings=None):
        if name_mappings is None:
            name_mappings = []
        self.name_mappings = name_mappings
        self.tools = self._generate_tools()

    def _method_from_tool(self, tool):
        method_name = tool
        for m, t in self.name_mappings:
            if t == tool:
                method_name = m
        method = getattr(self, method_name, None)
        # if not callable(method):
        #    raise TypeError(f"Attribute '{method_name}' is not callable.")
        return method

    def _generate_tools(self):
        functions = []
        methods = inspect.getmembers(self, predicate=inspect.ismethod)
        for name, method in methods:
            if name.startswith('_') or name == 'process':
                continue
            functions.append(method)
        generator = SchemaGenerator(name_mappings=self.name_mappings)
        tools = generator.generate_tools(*functions)
        return tools


    class Finish(BaseModel):
        reason: str = Field(description="The reasoning behind the answer")
        answer: str = Field(description="The answer to the user's question")

    def finish(self, param: Finish):
        """
        Finish the task and return the answer.
        """
        answer = param.answer
        if answer.lower() == 'yes' or answer.lower() == 'no':
            answer = answer.lower()
        return answer

    def process(self, function_call):
        tool_args = json.loads(function_call.arguments)
        tool_name = function_call.name
        return self._process_unpacked(tool_name, tool_args)

    def _process_unpacked(self, tool_name, tool_args):
        method = self._method_from_tool(tool_name)
        if method is None:
            return ToolResult(tool_name=tool_name, tool_args=tool_args, error=f"Unknown tool name: {tool_name}")
        parameters = inspect.signature(method).parameters
        if len(parameters) > 1:
            raise TypeError(f"Function {method.__name__} has more than one parameter")
        if len(parameters) == 1:
            for name, param in parameters.items():
                param_class = param.annotation
            if not issubclass(param_class, BaseModel):
                raise TypeError(f"The only parameter of method {method.__name__} is not a subclass of pydantic BaseModel")
            param = param_class(**tool_args)
            observations = method(param)
        else:
            observations = method()
        return ToolResult(tool_name=tool_name, tool_args=tool_args, observations=observations)


class WikipediaSearch(ToolBox):
    def __init__(self, wiki_api, cached=False):
        self.wiki_api = wiki_api
        self.cached = cached
        self.document = None
        name_mappings = [("next_lookup", "next")]
        super().__init__(name_mappings=name_mappings)

    class Search(BaseModel):
        reason: str = Field(description="The reason for searching")
        query: str = Field(description="The query to search for on Wikipedia")

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
        query: str = Field(description="The wikipedia page title")
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
            observations = observations + "The retrieved page summary starts with:\n" + document.first_chunk() + "\n"
        return observations

import json
import inspect
import docutils
from sphinx.ext.napoleon import Config, GoogleDocstring, NumpyDocstring
from docutils.parsers.rst import Parser
from docutils.utils import new_document
from typing import Annotated

from tool_def_generator import ToolDefGenerator


from answerbot.get_wikipedia import ContentRecord

class ToolResult:
    def __init__(self, tool_name=None, tool_args=None, observations=None, error=None):
        self.tool_name = tool_name
        self.tool_args = tool_args
        self.observations = observations
        self.error = error

class ToolBox:

    def __init__(self, reverse_function_mapping=None):
        self.function_mapping = { "finish": self.finish }
        if reverse_function_mapping is None:
            reverse_function_mapping = { "finish": "finish" }
        self.reverse_function_mapping = reverse_function_mapping
        self.tools = self._generate_tools()

    def _generate_tools(self):
        functions = []
        methods = inspect.getmembers(self, predicate=inspect.ismethod)
        generator_mapping = []
        for name, mapped_name in self.reverse_function_mapping.items():
            generator_mapping.append((name, mapped_name))

        for name, method in methods:
            if name.startswith('_') or name == 'process':
                continue
            functions.append(method)
        generator = ToolDefGenerator(name_mappings=generator_mapping)
        tools = generator.generate(*functions)
        return tools

    def finish(self, answer: Annotated[str, "The answer to the user's question"], reason: Annotated[str, "The reasoning behind the answer"]):
        """
        Finish the task and return the answer.
        """
        answer = answer
        if answer.lower() == 'yes' or answer.lower() == 'no':
            answer = answer.lower()
        return answer

    def process(self, function_call):
        tool_args = json.loads(function_call.arguments)
        tool_name = function_call.name
        if tool_name not in self.function_mapping:
            return ToolResult(tool_name=tool_name, tool_args=tool_args, error=f"Unknown tool name: {tool_name}")
        observations = self.function_mapping[tool_name](**tool_args)
        return ToolResult(tool_name=tool_name, tool_args=tool_args, observations=observations)


class WikipediaSearch(ToolBox):
    def __init__(self, wiki_api, cached=False):
        reverse_function_mapping = {
            "finish": "finish",
            "search": "search",
            "get": "get",
            "lookup": "lookup",
            "next_lookup": "next",
        }
        super().__init__(reverse_function_mapping=reverse_function_mapping)
        self.wiki_api = wiki_api
        self.cached = cached
        self.document = None
        self.function_mapping.update({
            "search": self.search,
            "get": self.get,
            "lookup": self.lookup,
            "next": self.next_lookup,
        })


    def search(self, query: Annotated[str, "The query to search for on Wikipedia"], reason: Annotated[str, "The reason for searching"]):
        """
        Searches Wikipedia, saves the first result page, and informs about the content of that page.
        """
        if not self.cached:
            search_query = query
            search_record = self.wiki_api.search(search_query)
        else:
            title = query
            chunk_size = self.wiki_api.chunk_size
            search_record = ContentRecord.load_from_disk(title, chunk_size)
        self.document = search_record.document
        return self._retrieval_observations(search_record)

    def get(self, title: Annotated[str, "The page title"], reason: Annotated[str, "The reason for retrieving the page"]):
        """
        Retrieves a Wikipedia page, saves the result, and informs about the content of that page.
        """

        if self.cached:
            raise Exception("Cached get not implemented")
        search_record = self.wiki_api.get_page(title)
        self.document = search_record.document
        return self._retrieval_observations(search_record)

    def lookup(self, keyword: Annotated[str, "The keyword to search"], reason: Annotated[str, "The reason for searching"]):
        """
        Looks up a word on the current page.
        """
        if self.document is None:
            observations = "No document defined, cannot lookup"
        else:
            text = self.document.lookup(keyword)
            observations = 'Keyword "' + keyword + '" '
            if text:
                num_of_results = len(self.document.lookup_results)
                observations = observations + f"found on current page in {num_of_results} places. The first occurence:\n" + text
            else:
                observations = observations + "not found in current page"
        return observations

    def next_lookup(self, reason: Annotated[str, "The reason for searching"]):
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

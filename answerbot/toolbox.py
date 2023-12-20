import json
import inspect
import docutils
from sphinx.ext.napoleon import Config, GoogleDocstring, NumpyDocstring
from docutils.parsers.rst import Parser
from docutils.utils import new_document

from .get_wikipedia import ContentRecord

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
        self.functions = []
        for function_info in self._generate_function_data(self.reverse_function_mapping):
            self.functions.append(function_info)

    def finish(self, answer, reason):
        """
        Finish the task and return the answer.

        :param answer: The answer to the user's question.
        :type answer: str
        :param reason: The reasoning behind the answer.
        :type reason: str
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

    def _generate_function_data(self, reverse_function_mapping, docstring_type='reStructuredText'):
        functions_data = []
        methods = inspect.getmembers(self, predicate=inspect.ismethod)
        for name, method in methods:
            if name.startswith('_') or name == 'process':
                continue

            docstring = method.__doc__
            if not docstring:
                continue

            try:
                api_name = reverse_function_mapping[name]
                function_info = self._parse_docstring(api_name, docstring, docstring_type)
            except ValueError as e:
                raise ValueError(f"Validation error in method '{name}': {e}")

            functions_data.append(function_info)

        return functions_data

    def _parse_docstring(self, api_name, docstring, style):
        if not docstring:
            raise ValueError("Missing docstring.")
        if not style in ['reStructuredText', 'rest', 'google', 'eval']:
            raise ValueError(f"Invalid docstring style: {style}")

        napoleon_config = Config(napoleon_use_param=True, napoleon_use_rtype=True)

        # Convert Google and NumPy docstrings to reST format
        if style == 'eval':
            if not docstring.startswith('{'):
                docstring = '{\n' + docstring + '\n}'
            function_info = eval(docstring)
            return function_info
        elif style in ['google', 'numpy']:
            if style == 'google':
                parsed_docstring = GoogleDocstring(docstring, napoleon_config)
            else:  # numpy
                parsed_docstring = NumpyDocstring(docstring, napoleon_config)

            # Convert to reST format
            rest_docstring = str(parsed_docstring)
        else:
            rest_docstring = str(docstring)

        description, parameters, required = self._parse_rtc_docstring(rest_docstring)
        function_info = {
            "name": api_name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required
            }
        }
        return function_info



    def _parse_rtc_docstring(self, docstring):
        # Create a new document for parsing
        settings = docutils.frontend.get_default_settings(Parser)
        document = new_document('<docstring>', settings)

        # Parse the docstring
        parser = Parser()
        parser.parse(docstring, document)

        params_dict = {}
        required_params = []
        description = ''

        # Traverse the document to find parameter descriptions
        for node in document.findall():
            # Extract method description from the first paragraph
            if isinstance(node, docutils.nodes.paragraph) and not description:
                description = node.astext()

            # Extract parameters from field lists
            if isinstance(node, docutils.nodes.field_list):
                for field_node in node:
                    if isinstance(field_node, docutils.nodes.field):
                        name_node, body_node = field_node.children
                        if isinstance(name_node, docutils.nodes.field_name) and isinstance(body_node, docutils.nodes.field_body):
                            (field_type, param_name) = name_node.astext().split()
                            if field_type == 'param':
                                param_desc = body_node.astext()
                                params_dict[param_name] = {'description': param_desc}
                                required_params.append(param_name)
                            elif field_type == 'type':
                                param_type = body_node.astext()
                                if param_type == 'str':
                                    param_type = 'string'
                                params_dict[param_name]['type'] = param_type

            # Extract parameters from definition lists
            elif isinstance(node, docutils.nodes.definition_list):
                for definition_item in node:
                    if isinstance(definition_item, docutils.nodes.definition_list_item):
                        term, definition = definition_item.children
                        term_text = term.astext()
                        # Check if the term is about parameters (e.g., "Args:")
                        if term_text.lower() in ['args', 'args:', 'parameters', 'parameters:']:
                            for params_node in definition.findall(docutils.nodes.paragraph):
                                print(params_node)
                                param_text = params_node.children[0].astext()
                                params_lines = param_text.splitlines()
                                for param_line in params_lines:
                                    param_term, param_desc = param_line.split(':', 1)
                                    param_desc = param_desc.strip()
                                    param_name, param_type = param_term.split()
                                    param_name = param_name.strip()
                                    param_type = param_type.strip('()')
                                    if param_type == 'str':
                                        param_type = 'string'
                                    params_dict[param_name] = {'description': param_desc, 'type': param_type}
                                    required_params.append(param_name)

        return description, params_dict, required_params


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



    def search(self, query, reason):
        """
        Searches Wikipedia, saves the first result page, and informs about the content of that page.

        :param query: The query to search for on Wikipedia.
        :type query: str
        :param reason: The reason for searching.
        :type reason: str
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

    def get(self, title, reason):
        """
        Retrieves a Wikipedia page, saves the result, and informs about the content of that page.

        :param title: The query to search for on Wikipedia.
        :type title: str
        :param reason: The reason for getting the page.
        :type reason: str
        """

        if self.cached:
            raise Exception("Cached get not implemented")
        search_record = self.wiki_api.get_page(title)
        self.document = search_record.document
        return self._retrieval_observations(search_record)

    def lookup(self, keyword, reason):
        """
        Looks up a word on the current page.

        :param keyword: The keyword to search for on current page.
        :type keyword: str
        :param reason: The reason for searching.
        :type reason: str
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

    def next_lookup(self, reason):
        """
        Jumps to the next occurrence of the word searched previously.

        :param reason: The reason for searching.
        :type reason: str
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

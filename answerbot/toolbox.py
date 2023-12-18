import json
from .get_wikipedia import ContentRecord

class ToolResult:
    def __init__(self, tool_name=None, tool_args=None, observations=None, error=None):
        self.tool_name = tool_name
        self.tool_args = tool_args
        self.observations = observations
        self.error = error

class ToolBox:

    def __init__(self):
        self.function_mapping = { "finish": self.finish }
        self.functions = []

    def finish(self, tool_args):
        answer = tool_args["answer"]
        if answer.lower() == 'yes' or answer.lower() == 'no':
            answer = answer.lower()
        return answer

    def process(self, function_call, **kwargs):
        tool_args = json.loads(function_call.arguments)
        tool_name = function_call.name
        if tool_name not in self.function_mapping:
            return ToolResult(tool_name=tool_name, tool_args=tool_args, error=f"Unknown tool name: {tool_name}")
        observations = self.function_mapping[tool_name](tool_args, **kwargs)
        return ToolResult(tool_name=tool_name, tool_args=tool_args, observations=observations)


class WikipediaSearch(ToolBox):
    def __init__(self, wiki_api):
        super().__init__()
        self.wiki_api = wiki_api
        self.document = None
        self.function_mapping.update({
            "search": self.search,
            "get": self.get,
            "lookup": self.lookup,
            "next": self.next_lookup,
        })

        self.functions = [
            {
                "name": "search",
                "description": "searches Wikipedia saves the first result page and informs about the content of that page",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The query to search for on Wikipedia",
                        },
                        "reason": {
                            "type": "string",
                            "description": "The reason for searching",
                        }
                    },
                    "required": ["query", "reason"],
                },
            },
            {
                "name": "get",
                "description": "gets the Wikipedia page with the given title, saves it and informs about the content of that page",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "The title of the Wikipedia page to get",
                        },
                        "reason": {
                            "type": "string",
                            "description": "The reason for retrieving this particular article",
                        }
                    },
                    "required": ["title", "reason"],
                },
            },
            {
                "name": "lookup",
                "description": "returns text surrounding the keyword in the current page",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keyword": {
                            "type": "string",
                            "description": "The keyword or section to lookup within the retrieved content",
                        },
                        "reason": {
                            "type": "string",
                            "description": "The reason for checking in this particular section of the article",
                        }
                    },
                    "required": ["keyword", "reason"],
                },
            },
            {
                "name": "next",
                "description": "returns next occurrence of the looked up keyword in the current page",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {
                            "type": "string",
                            "description": "The reason for checking in this particular section of the article",
                        }
                    },
                    "required": ["reason"],
                },
            },
            {
                "name": "finish",
                "description": "Finish the task and return the answer",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "description": "The answer to the user's question",
                        },
                        "reason": {
                            "type": "string",
                            "description": "The reasoning behind the answer",
                        }
                    },
                    "required": ["answer", "reason"],
                },
            },
        ]

    def search(self, function_args, cached=False):
        if not cached:
            search_query = function_args["query"]
            search_record = self.wiki_api.search(search_query)
        else:
            title = function_args["query"]
            chunk_size = self.wiki_api.chunk_size
            search_record = ContentRecord.load_from_disk(title, chunk_size)
        self.document = search_record.document
        return self.retrieval_observations(search_record)

    def get(self, function_args, cached=False):
        if cached:
            raise Exception("Cached get not implemented")
        search_record = self.wiki_api.get_page(function_args["title"])
        self.document = search_record.document
        return self.retrieval_observations(search_record)

    def lookup(self, function_args):
        keyword = function_args["keyword"]
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

    def next_lookup(self, function_args):
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

    def retrieval_observations(self, search_record, limit_sections=None):
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

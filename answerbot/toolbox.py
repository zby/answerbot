from .get_wikipedia import ContentRecord


class ToolBox:
    def __init__(self):
        self.function_mapping = {}
        self.functions = []

    def process(self, function_name, function_args, **kwargs):
        if function_name not in self.function_mapping:
            print(f"<<< Unknown function name: {function_name}")
            raise Exception(f"Unknown function name: {function_name}")
        return self.function_mapping[function_name](function_args, **kwargs)


class WikipediaSearch(ToolBox):
    def __init__(self, wiki_api):
        super().__init__()
        self.wiki_api = wiki_api
        self.document = None
        self.function_mapping = {
            "search": self.search,
            "get": self.get,
            "lookup": self.lookup,
            "next": self.next_lookup,
        }

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
                observations = observations + f"found on current page in in {num_of_results} places. The first occurence:\n" + text
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

import openai
import json
import pprint

from get_wikipedia import WikipediaApi

from prompt_builder import FunctionalPrompt, PlainTextPrompt, System, Assistant, FunctionCall, FunctionResult
from react_prompt import Question, FunctionalReactPrompt, TextReactPrompt, retrieval_observations, lookup_observations



functions = [
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
                "thought": {
                    "type": "string",
                    "description": "The reason for searching",
                }
            },
            "required": ["query", "thought"],
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
                "thought": {
                    "type": "string",
                    "description": "The reason for retrieving this particular article",
                }
            },
            "required": ["query", "thought"],
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
                "thought": {
                    "type": "string",
                    "description": "The reason for checking in this particular section of the article",
                }
            },
            "required": ["keyword", "thought"],
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
                }
            },
            "required": ["answer"],
        },
    },
]

function_names = [f["name"] for f in functions]

def openai_query(prompt, model):
    def convert_to_dict(obj):
        if isinstance(obj, openai.openai_object.OpenAIObject):
            return {k: convert_to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_dict(item) for item in obj]
        else:
            return obj

    args = {}
    if isinstance(prompt, FunctionalPrompt):
        args["functions"] = functions
        args["function_call"] = "auto"
    else:
        args["stop"] = ["\nObservation:"]

    openai.api_requestor.TIMEOUT_SECS = 60
    response = openai.ChatCompletion.create(
        model=model,
        messages=prompt.to_messages(),
        **args
    )

    response_message = response["choices"][0]["message"]
    return convert_to_dict(response_message)

def function_call_from_functional(response):
    return response.get("function_call")

def function_call_from_plain(response):
    response_lines = response["content"].strip().split('\n')
    if len(response_lines) >= 2:
        last_but_one_line = response_lines[-2]
    else:
        last_but_one_line = ""
    last_line = response_lines[-1]
    if last_line.startswith('Action: '):
        if last_but_one_line.startswith('Thought: '):
            thought = last_but_one_line[9:]
        else:
            thought = None
        if last_line.startswith('Action: finish['):
            answer = last_line[15:-1]
            return {
                "name": "finish",
                "arguments": json.dumps({"answer": answer, "thought": thought})
            }
        elif last_line.startswith('Action: search['):
            query = last_line[15:-1]
            return {
                "name": "search",
                "arguments": json.dumps({"query": query, "thought": thought})
            }
        elif last_line.startswith('Action: get['):
            query = last_line[15:-1]
            return {
                "name": "get",
                "arguments": json.dumps({"title": query, "thought": thought})
            }
        elif last_line.startswith('Action: lookup['):
            keyword = last_line[15:-1]
            return {
                "name": "lookup",
                "arguments": json.dumps({"keyword": keyword, "thought": thought})
            }

    return None

def run_conversation(prompt, config):
    document = None
    wiki_api = WikipediaApi(max_retries=3, chunk_size=config['chunk_size'])
    iter = 0
    while True:
        print(f">>>Iteration: {iter}")
        response = openai_query(prompt, config['model'])
        if config['functional']:
            function_call = function_call_from_functional(response)
        else:
            function_call = function_call_from_plain(response)
        #print("model response: ", response)

        # Process the function calls
        if function_call:
            function_name = function_call["name"]
            if function_name not in function_names:
                print(f"<<< Unknown function name: {function_name}")
                raise Exception(f"Unknown function name: {function_name}")
            function_args = json.loads(function_call["arguments"])
            # sometimes model returns a function call without a thought
            # or alternatively a function call with a thought but without a content
            if 'thought' not in function_args and 'content' in response:   #
                function_args['thought'] = response['content']
            message = FunctionCall(function_name, **function_args)
            print("<<< ", message.plaintext())
            prompt.push(message)
            if function_name == "finish":
                answer = function_args["answer"]
                # normalize the answer
                if answer.lower() == 'yes' or answer.lower == 'no':
                    answer = answer.lower()
                return answer, prompt
            elif function_name == "search":
                search_record = wiki_api.search(function_args["query"])
                document = search_record.document
                observations = retrieval_observations(search_record)
            elif function_name == "get":
                search_record = wiki_api.get_page(function_args["title"])
                document = search_record.document
                observations = retrieval_observations(search_record)
            elif function_name == "lookup":
                observations = lookup_observations(document, function_args["keyword"])
            message = FunctionResult(function_name, observations)
        else:
            message = Assistant(response.get("content"))
        print("<<< ", message.plaintext())
        prompt.push(message)
        iter = iter + 1
        if iter >= config['max_llm_calls']:
            print("<<< Max llm calls reached, exiting.")
            return None, prompt

def get_answer(question, config):
    print("\n\n<<< Question:", question)
    # Check that config contains the required fields
    required_fields = ['chunk_size', 'functional', 'example_chunk_size', 'max_llm_calls', ]
    missing_fields = [field for field in required_fields if field not in config]
    if missing_fields:
        raise ValueError(f"Missing required config fields: {', '.join(missing_fields)}")

    if config['functional']:
        prompt = FunctionalReactPrompt(question, config['example_chunk_size'])
    else:
        prompt = TextReactPrompt(question, config['example_chunk_size'])
    return run_conversation(prompt, config)

if __name__ == "__main__":
    # load the api key from a file
    with open("config.json", "r") as f:
        json_config = json.load(f)
    openai.api_key = json_config["api_key"]

    # question = "What was the first major battle in the Ukrainian War?"
    # question = "What were the main publications by the Nobel Prize winner in economics in 2023?"
    # question = "What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?"
    # question = 'Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?'
    # question = "how old was Donald Tusk when he died?"
    # question = "how many keys does a US-ANSI keyboard have on it?"
    # question = "How many children does Donald Tusk have?"
    # question = "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?"
    # question = "The director of the romantic comedy \"Big Stone Gap\" is based in what New York city?"
    question = "The arena where the Lewiston Maineiacs played their home games can seat how many people?"

    config = {
        "chunk_size": 300,
        "functional": False,
        "example_chunk_size": 300,
        "max_llm_calls": 5,
        "model": "gpt-3.5-turbo",
    }

    answer, prompt = get_answer(question, config)
    print(prompt.to_text())

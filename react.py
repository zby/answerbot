import openai
import json
import pprint

from get_wikipedia import WikipediaApi

from prompt_builder import Prompt, PromptMessage, OpenAIMessage, User, System, Assistant, FunctionCall, FunctionResult
from react_prompt import system_message, get_examples, retrieval_observations, lookup_observations

MAX_ITER = 5
CHUNK_SIZE = 400
FUNCTIONAL_STYLE = False
MODEL = "gpt-3.5-turbo-0613"

functions = [
    {
        "name": "search",
        "description": "Search Wikipedia for a query, retrieve the page, save it for later and return the summary",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to search for on Wikipedia",
                },
                "thought": {
                    "type": "string",
                    "description": "The reason for searching for this particular article",
                }
            },
            "required": ["query", "thought"],
        },
    },
    {
        "name": "lookup",
        "description": "Look up a word in the saved Wikipedia page and return text surrounding it",
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

def openai_query(messages, functions=None):
    def convert_to_dict(obj):
        if isinstance(obj, openai.openai_object.OpenAIObject):
            return {k: convert_to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_dict(item) for item in obj]
        else:
            return obj

    args = { 'stop': ["\nObservation:"] }
    if functions is not None:
        args["functions"] = functions
        args["function_call"] = "auto"

    if type(messages) == str:
        messages = [{ "role": "user", "content": messages }]

    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=messages,
        **args
    )

    response_message = response["choices"][0]["message"]
    return convert_to_dict(response_message)

def function_call_from_functional(response):
    return response.get("function_call")

def function_call_from_plain(response):
    response_lines = response["content"].strip().split('\n')
    last_line = response_lines[-1]

    if last_line.startswith('Action: finish['):
        answer = last_line[15:-1]
        return {
            "name": "finish",
            "arguments": json.dumps({"answer": answer})
        }
    elif last_line.startswith('Action: search['):
        query = last_line[15:-1]
        return {
            "name": "search",
            "arguments": json.dumps({"query": query})
        }
    elif last_line.startswith('Action: lookup['):
        keyword = last_line[15:-1]
        return {
            "name": "lookup",
            "arguments": json.dumps({"keyword": keyword})
        }
    else:
        return None

def run_conversation(prompt, chunk_size, functional):
    document = None
    wiki_api = WikipediaApi(max_retries=3, chunk_size=chunk_size)
    iter = 0
    while True:
        print(f">>>Iteration: {iter}")
        if functional:
            response = openai_query(prompt.openai_messages(), functions)
            function_call = function_call_from_functional(response)
        else:
            response = openai_query(prompt.plain())
            function_call = function_call_from_plain(response)
        print("model response: ", response)

        # Process the function calls
        if function_call:
            function_name = function_call["name"]
            if function_name not in function_names:
                print(f"<<< Unknown function name: {function_name}")
                raise Exception(f"Unknown function name: {function_name}")
            function_args = json.loads(function_call["arguments"])
            message = FunctionCall(function_name, **function_args)
            print("<<< ", message.plaintext())
            prompt.push(message)
            if function_name == "finish":
                answer = function_args["answer"]
                return answer
            elif function_name == "search":
                search_record = wiki_api.search(function_args["query"])
                document = search_record.document
                observations = retrieval_observations(search_record)
            elif function_name == "lookup":
                observations = lookup_observations(document, function_args["keyword"])
            message = FunctionResult(function_name, observations)
            print("<<< ", message.plaintext())
            prompt.push(message)
        else:
            prompt.push(Assistant(response.get("content")))
        iter = iter + 1
        if iter >= MAX_ITER:
            print("<<< Max iterations reached, exiting.")
            return None

def get_answer(question, chunk_size, functional):
    examples = get_examples()
    prompt = Prompt([
        system_message,
        *examples,
        User(f"Question: {question}"),
    ])
    #print(prompt.plain())
    return run_conversation(prompt, chunk_size, functional)

if __name__ == "__main__":
    # load the api key from a file
    with open("config.json", "r") as f:
        config = json.load(f)
    openai.api_key = config["api_key"]

    # question = "What was the first major battle in the Ukrainian War?"
    # question = "What were the main publications by the Nobel Prize winner in economics in 2023?"
    # question = "What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?"
    # question = 'Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?'
    # question = "how old was Donald Tusk when he died?"
    question = "how many keys does a US-ANSI keyboard have on it?"
    # question = "How many children does Donald Tusk have?"

    result = get_answer(question, CHUNK_SIZE, FUNCTIONAL_STYLE)
    print(result)

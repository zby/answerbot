import openai
import json

from get_wikipedia import WikipediaApi

MAX_ITER = 3

system_message = '''You are a helpful AI assistant trying to answer questions.
When you have enough information to answer the question please call the finish function with the answer.
When you need additional information please use the available functions to get it.
After each function call, please analyze the response reflect on it and decide what to do next.
'''

def run_conversation(question):
    document = None
    wiki_api = WikipediaApi(max_retries=3)
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": question}
    ]
    functions = [
        {
            "name": "search_wikipedia",
            "description": "Search Wikipedia for a query, retrieve the page, save it for later and return the summary",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to search for on Wikipedia",
                    }
                },
                "required": ["query"],
            },
        },
        {
            "name": "lookup_word",
            "description": "Look up a word in the saved Wikipedia page and return text surrounding it",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "The keyword to lookup within the retrieved content",
                    }
                },
                "required": ["keyword"],
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

    iter = 0
    while True:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=messages,
            functions=functions,
            function_call="auto",
        )

        response_message = response["choices"][0]["message"]

        if response_message.get("function_call"):
            function_name = response_message["function_call"]["name"]
            function_args = json.loads(response_message["function_call"]["arguments"])
            if function_name == "finish":
                answer = function_args["answer"]
                return answer
            elif function_name == "search_wikipedia":
                search_record = wiki_api.search(function_args["query"])
                document = search_record.document
                function_response = ''
                for record in search_record.retrieval_history:
                    print(">>>", record)
                    function_response = function_response + record + "\n"
                function_response = "Successfully retrieved " + function_args["query"] + " from Wikipedia.\n"
                function_response = function_response + "The retrieved wikipedia page summary contains: " + document.first_chunk() + "\n"
            elif function_name == "lookup_word":
                if document is None:
                    function_response = "No document defined, cannot lookup"
                text = document.lookup(function_args["keyword"])
                if text:
                    function_response = f"Found keyword: \n{text}"
                else:
                    function_response = 'Keyword "' + function_args["keyword"] + '" not found in current page'

            messages.append(response_message)
            messages.append(
                {
                    "role": "function",
                    "name": function_name,
                    "content": function_response,
                }
            )
    iter = iter + 1
    if iter >= MAX_ITER:
        print("Max iterations reached, exiting.")
        return None


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
    # question = "how many keys does a US-ANSI keyboard have on it?"
    question = "How many children does Donald Tusk have?"
    question = "What were the main publications by the Nobel Prize winner in economics in 2023?"
    result = run_conversation(question)
    print(result)

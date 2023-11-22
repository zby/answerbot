import openai
import json
import time

from get_wikipedia import WikipediaApi

from prompt_builder import FunctionalPrompt, Assistant, FunctionCall, FunctionResult
from react_prompt import FunctionalReactPrompt, NewFunctionalReactPrompt, TextReactPrompt, ToolBox

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

    errors = []
    for i in range(2):
        try:
            openai.api_requestor.TIMEOUT_SECS = i * 20 + 20
            response = openai.ChatCompletion.create(
                model=model,
                messages=prompt.to_messages(),
                **args
            )
            break
        except openai.error.Timeout as e:
            print("OpenAI Timeout: ", e)
            time.sleep(20)
            errors.append(e)
            continue
        except openai.error.APIError as e:
            print("OpenAI APIError: ", e)
            time.sleep(20)
            errors.append(e)
            continue
    if response is None:
        errors_string = "\n".join([str(e) for e in errors])
        raise Exception(f"OpenAI API calls failed: {errors_string}")
    response_message = response["choices"][0]["message"]
    return convert_to_dict(response_message)


def run_conversation(prompt, config, toolbox):
   iter = 0
   while True:
       print(f">>>LLM call number: {iter}")
       response = openai_query(prompt, config['model'])
       print("<<< ", response)
       function_call = prompt.function_call_from_response(response)

       # Process the function calls
       if function_call:
           print("function_call: ", function_call)
           message = FunctionCall(function_call["name"], **json.loads(function_call["arguments"]))
           prompt.push(message)

           result = toolbox.process(function_call)
           print("<<< ", result)
           message = FunctionResult(function_call["name"], result)
           prompt.push(message)
           if toolbox.answer is not None:
               print("<<< Conversation finished.")
               return toolbox.answer, prompt
       else:
           message = Assistant(response.get("content"))
           prompt.push(message)
       if iter >= config['max_llm_calls']:
           print("<<< Max llm calls reached, exiting.")
           return None, prompt
       iter = iter + 1
       time.sleep(60)


def get_answer(question, config):
    print("\n\n<<< Question:", question)
    # Check that config contains the required fields
    required_fields = ['chunk_size', 'prompt', 'example_chunk_size', 'max_llm_calls', ]
    missing_fields = [field for field in required_fields if field not in config]
    if missing_fields:
        raise ValueError(f"Missing required config fields: {', '.join(missing_fields)}")
    # A dictionary that maps class names to classes
    CLASS_MAP = {
        'NFRP': NewFunctionalReactPrompt,
        'FRP': FunctionalReactPrompt,
        'TRP': TextReactPrompt
    }
    prompt_class = CLASS_MAP[config['prompt']]
    prompt = prompt_class(question, config['example_chunk_size'])
    wiki_api = WikipediaApi(max_retries=3, chunk_size=config['chunk_size'])
    toolbox = ToolBox(wiki_api)
    return run_conversation(prompt, config, toolbox)

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
    #question = "When Poland became elective monarchy?"
    question = "Were Scott Derrickson and Ed Wood of the same nationality?"
    question = "What science fantasy young adult series, told in first person, has a set of companion books narrating the stories of enslaved worlds and alien species?"
    question = "The arena where the Lewiston Maineiacs played their home games can seat how many people?"
    question = "What is the name of the fight song of the university whose main campus is in Lawrence, Kansas and whose branch campuses are in the Kansas City metropolitan area?"
    question = "What year did Guns N Roses perform a promo for a movie starring Arnold Schwarzenegger as a former New York Police detective?"

    config = {
        "chunk_size": 300,
        "prompt": 'NFRP',
        "example_chunk_size": 200,
        "max_llm_calls": 7,
        "model": "gpt-4",
    }

    answer, prompt = get_answer(question, config)
    print(prompt.to_text())

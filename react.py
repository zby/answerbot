import openai
import requests
import json
import time

import get_wikipedia
from get_wikipedia import WikipediaApi, ContentRecord
from reactors import FunctionalReactor, TextualReactor

def openai_query(messages, functions=None):
    def convert_to_dict(obj):
        if isinstance(obj, openai.openai_object.OpenAIObject):
            return {k: convert_to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_dict(item) for item in obj]
        else:
            return obj

    args = {}
    if functions is not None:
        args["functions"] = functions
        args["function_call"] = "auto"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        **args
    )

    response_message = response["choices"][0]["message"]
    return convert_to_dict(response_message)


def wikipedia_search(entity):
    wiki_api = WikipediaApi(max_retries=3)
    return wiki_api.search(entity)

def get_answer(config, question, reactor) -> (str, int):
    iter = 0
    document = None
    wiki_api = WikipediaApi(max_retries=3)
    reactor.add_question(question)

    while True:
        action = reactor.query()
        iter += 1

        if action is None:
            return None, iter

        if action.name == 'finish':
            print(f'Answer: {action.argument}')
            return action.argument, iter
        elif action.name == 'search':
            query = action.argument
            print(f'Will search wikipedia for "{query}"')
            search_record = wiki_api.search(query)
            document = search_record.document
            text = document.first_chunk()
            for record in search_record.retrieval_history:
                print(">>>", record)
                reactor.add_observation(record, action)

            summary = f'the retrieved wikipedia page summary contains: {text}'
            print(">>>", summary)
            reactor.add_observation(summary, action)

            sections = document.section_titles()
            sections_list_md = "\n".join(map(lambda section: f' - {section}', sections))
            observation = f'the retrieved page contains the following sections:\n{sections_list_md}'
            print(">>>", observation)
            reactor.add_observation(observation, action)
        elif action.name == 'lookup':
            query = action.argument
            if document is None:
                print("No document defined, cannot lookup")
                return None, iter
            print(f'Will lookup paragraph containing "{query}"')
            text = document.lookup(query)
            if text is None or text == '':
                text = f"{query} not found in document"
            print(">>>", text)
            reactor.add_observation(text, action)
        if iter >= config["max_iter"]:
            print("Max iterations reached, exiting.")
            return None, iter

# Example usage
if __name__ == "__main__":
    # load the api key from a file
    with open("config.json", "r") as f:
        config = json.load(f)
        openai.api_key = config["api_key"]
        config["max_iter"] = config.get("max_iter", 5)
    # question = "What was the first major battle in the Ukrainian War?"
    # question = "What were the main publications by the Nobel Prize winner in economics in 2023?"
    # question = "What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?"
    # question = 'Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?'
    # question = "how old was Donald Tusk when he died?"
    # question = "how many keys does a US-ANSI keyboard have on it?"
    question = "How many children does Donald Tusk have?"

    mode = 'compare'

    reactors = []
    if mode == 'textual' or mode == 'compare':
        reactors.append(TextualReactor(openai_query))
    if mode == 'functional' or mode == 'compare':
        reactors.append(FunctionalReactor(openai_query))

    for reactor in reactors:
        reactor_name = reactor.__class__.__name__
        print('Obtaining an answer using', reactor_name)
        answer, iterations = get_answer(config, question, reactor)
        answer = f'"answer"' if answer else "Answer was not"
        print(f'[{reactor_name}] {answer} found after {iterations} queries')

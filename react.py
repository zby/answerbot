import requests
import json
import time

import get_wikipedia
from get_wikipedia import WikipediaApi, ContentRecord
from textual_reactor import TextualReactor

def openai_query(messages, api_key):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-3.5-turbo-16k",
        "messages": messages,
        "temperature": 0.7,
        # "stop": stops,
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
    if response.status_code >= 500:
        print(f"OpenAI temporarily unavailable (code {response.status_code}), waiting a bit...")
        time.sleep(5)
        return openai_query(messages, api_key)
    if response.status_code == 429:
        # TODO: maybe parse out the "Please try again in Xms" part and be clever
        print("Rate limit reached, waiting a bit...")
        time.sleep(5)
        return openai_query(messages, api_key)
    response_json = response.json()
    print(response_json)
    return response_json["choices"][0]["message"]


def wikipedia_search(entity):
    wiki_api = WikipediaApi(max_retries=3)
    return wiki_api.search(entity)

def get_answer(config, question):
    api_key = config["api_key"]
    MAX_ITER = 5
    iter = 0
    document = None
    wiki_api = WikipediaApi(max_retries=3)
    reactor = TextualReactor(lambda messages: openai_query(messages, api_key))
    reactor.add_question(question)
    while True:
        (action, argument) = reactor.query()
        if action == 'finish':
            answer = argument
            print(f'Answer: {answer}')
            return answer
        elif action == 'search':
            query = argument
            print(f'Will search wikipedia for "{query}"')
            search_record = wiki_api.search(query)
            document = search_record.document
            text = document.first_chunk()
            for record in search_record.retrieval_history:
                print(">>>", record)
                reactor.add_observation(record)

            summary = f'the retrieved wikipedia page summary contains: {text}'
            print(">>>", summary)
            reactor.add_observation(summary)

            sections = document.section_titles()
            observation = f'the retrieved page contains the following sections: {", ".join(sections)}'
            print(">>>", observation)
            reactor.add_observation(observation)
        elif action == 'lookup':
            if document is None:
                print("No document defined, cannot lookup")
                return None
            print(f'Will lookup paragraph containing "{query}"')
            text = document.lookup(query)
            if text is None or text == '':
                text = f"{query} not found in document"
            reactor.add_observation(text)
        if iter >= MAX_ITER:
            print("Max iterations reached, exiting.")
            return None
        iter += 1

# Example usage
if __name__ == "__main__":
    # load the api key from a file
    with open("config.json", "r") as f:
        config = json.load(f)

    # question = "What was the first major battle in the Ukrainian War?"
    # question = "What were the main publications by the Nobel Prize winner in economics in 2023?"
    # question = "What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?"
    # question = 'Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?'
    question = "how old was Donald Tusk when he died?"
    # question = "how many keys does a US-ANSI keyboard have on it?"

    get_answer(config, question)




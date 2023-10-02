import requests
import json
import re
import sys
import time
from typing import Iterable

### Sad testcases:
# - question: "What was the first major battle in the Ukrainian War?"
#   wikipedia: Russian_invasion_of_Ukraine
#   problem: too many tokens, have to pick the best section

# - question: "how old was Donald Tusk when he died" (trick question)
#   wikipedia: Donald Tusk
#   problem: "wikipedia.exceptions.PageError: Page id "donald buck" does not match any pages.".
#             Wikipedia fetches a different page than it's asked to
#             (one of Ukrainian War questions also did this)

### Happy testcases

# - question: how many planets are in the solar system
#   wikipedia: Planets in the Solar System

# - question: "how many keys does a US-ANSI keyboard have on it"
#   wikipedia: British and American keyboards

# - question: "how dense is ceramic"
#   wikipedia: Ceramic
#   notes: Answer:
#       Ceramic materials are hard, brittle, and strong in compression, but weak in shearing and tension.
#       They can withstand high temperatures ranging from 1,000 °C to 1,600 °C (1,800 °F to 3,000 °F).
#   --> The density of ceramics is not mentioned in the given information.
#   !!!



import wikipedia

question = input("Enter your question: ")
#question = "What was the first major battle in the Ukrainian War?"

# load the api key from a file
with open("config.json", "r") as f:
    config = json.load(f)
    api_key = config["api_key"]

def concise_answer(prompt: str) -> str:
    return f"When answering the following question be concise - reply with only the text of the search term, don't repeat the question. {prompt}"

# Define a function to interact with OpenAI API
def openai_query(prompt, api_key):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-3.5-turbo-16k",
        "messages": [{ "role": "user", "content": prompt }],
        "temperature": 0.7,
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
    if response.status_code == 429:
        # TODO: maybe parse out the "Please try again in Xms" part and be clever
        print("Rate limit reached, waiting a bit...")
        time.sleep(5)
        return openai_query(prompt, api_key)
    response_json = response.json()
    return response_json["choices"][0]["message"]["content"].strip()

def get_sections(page: wikipedia.WikipediaPage) -> Iterable[str]:
    # theoretically `page.sections` should do this, but it doesn't
    sections = []
    for line in page.content.split('\n'):
        match = re.match(r'\s*(=+)\s*(.*?)\s*(\1)', line)
        if match:
            print(match)
            sections.append(match[2])
    return sections

prompt = f'What wikipedia article would you search for in order to find an answer to the following question: "{question}"?'
query = openai_query(concise_answer(prompt), api_key)

print(f'Will search wikipedia for "{query}"')

wikipages = wikipedia.search(query)
if len(wikipages) == 0:
    print('No relevant wikipedia articles found')
    sys.exit(0)

print(f'Pages found:', ", ".join(wikipages))
print(f'Getting the contents of "{wikipages[0]}"')

page = wikipedia.page(wikipages[0])
# print(page.content)

# Required when token length exceeded
# sections = get_sections(page)
# print(sections)
# section_list = "\n".join(list(map(lambda section: f' - {section}', sections)))
# 
# prompt = f'Given the following sections of a wikipedia page, which one would you choose to answer the question "{question}"\n\n{section_list}'
# response = openai_query(concise_answer(prompt), api_key)
# print(response)

prompt = f'Given the contents of the wikipedia page included below, answer the following question: "{question}"\n\n{page.content}'
response = openai_query(prompt, api_key)
print(response)

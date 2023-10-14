import requests
import json
import time

import wikipedia
from document import WikipediaDocument

cot_prompt_short = '''
Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.
After each observation, provide the next Thought and next Action. Here are some examples:

Question: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?
Thought: I need to search Colorado orogeny, find the area that the eastern sector of the Colorado orogeny extends into, then find the elevation range of the area.
Action: Search[Colorado orogeny]
Observation: The Colorado orogeny was an episode of mountain building (an orogeny) in Colorado and surrounding areas. This took place from 1780 to 1650 million years ago (Mya), during the Paleoproterozoic (Statherian Period). It is recorded in the Colorado orogen, a >500-km-wide belt of oceanic arc rock that extends southward into New Mexico. The Colorado orogeny was likely part of the larger Yavapai orogeny.
Thought: It does not mention the eastern sector of the Colorado orogeny. I need to look up eastern sector.
Action: Lookup[eastern sector]
Observation: The Colorado orogen, formerly called the Colorado province, is a >500-km-wide belt of oceanic arc rock (1.78\u20131.65 Ga) that extends southward into New Mexico and composes a major part of the Proterozoic provinces of southwestern United States. This transcontinental collisional event occurred during the  Paleoproterozoic (Statherian Period).[1] The Wyoming sector of the Colorado orogeny was formerly called the Medicine Bow orogeny. The eastern sector extends into the High Plains and is called the Central Plains orogeny. The boundary between the Colorado orogeny and the Wyoming craton is the Cheyenne belt, a 5-km-wide mylonitic shear zone that verges northward. The Cheyenne belt transects and cuts off the south edge of the older Trans-Hudson orogeny.[2]
Thought: The eastern sector of Colorado orogeny extends into the High Plains, so High Plains is the area. I need to search High Plains and find its elevation range.
Action: Search[High Plains]Observation 3: High Plains refers to one of two distinct land regions:
Thought: I need to search High Plains (United States).
Action: Search[High Plains (United States)]
Observation 4: The High Plains are a subregion of the Great Plains, mainly in the Western United States, but also partly in the Midwest states of Nebraska, Kansas, and South Dakota, generally encompassing the western part of the Great Plains before the region reaches the Rocky Mountains. The High Plains are located in eastern Montana, southeastern Wyoming, southwestern South Dakota, western Nebraska, eastern Colorado, western Kansas, eastern New Mexico, western Oklahoma, and to just south of the Texas Panhandle.[4] The southern region of the Western High Plains ecology region contains the geological formation known as Llano Estacado which can be seen from a short distance or on satellite maps.[5] From east to west, the High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130 m).[3]
Thought: High Plains rise in elevation from around 1,800 to 7,000 ft, so the answer is 1,800 to 7,000 ft.
Action: Finish[1,800 to 7,000 ft]

Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
Thought: The question simplifies to "The Simpsons" character Milhouse is named after who. I only need to search Milhouse and find who it is named after.
Action: Search[Milhouse]
Observation: Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening.
Thought: The paragraph does not tell who Milhouse is named after, maybe I can look up "named after".
Action: Lookup[named after]
Observation: Milhouse was named after U.S. president Richard Nixon, whose middle name was Milhous. 
Thought: Milhouse was named after U.S. president Richard Nixon, so the answer is Richard Nixon.
Action: Finish[President Richard Nixon]

{input}
'''


def openai_query(prompt, stops, api_key):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-3.5-turbo-16k",
        "messages": [{ "role": "user", "content": prompt }],
        "temperature": 0.7,
        "stop": stops,
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
    if response.status_code == 429:
        # TODO: maybe parse out the "Please try again in Xms" part and be clever
        print("Rate limit reached, waiting a bit...")
        time.sleep(5)
        return openai_query(prompt, stops, api_key)
    response_json = response.json()
    return response_json["choices"][0]["message"]["content"].strip()


# load the api key from a file
with open("config.json", "r") as f:
    config = json.load(f)
    api_key = config["api_key"]

question = "What was the first major battle in the Ukrainian War?"

def wikipedia_search(entity):
    wikipages = wikipedia.search(entity)
    if len(wikipages) == 0:
        print('No relevant wikipedia articles found')
        sys.exit(0)

    print(f'Pages found:', ", ".join(wikipages))
    print(f'Getting the contents of "{wikipages[0]}"')

    page = None

    for result in wikipages:
        try:
            page = wikipedia.page(result).content
            break
        except wikipedia.DisambiguationError as de:
            # Handle disambiguation pages by attempting to get the first option
            try:
                page = wikipedia.page(de.options[0]).content
                break
            except:
                continue
        except wikipedia.exceptions.PageError:
            # If the page doesn't exist, move to the next result
            continue
    return WikipediaDocument(page)


done = False
MAX_ITER = 2
iter = 0
prompt = cot_prompt_short.format(input=question)
document = None
while not done:
    print(prompt)
    reaction = openai_query(prompt, "\nObservation:", api_key)
    print(reaction)
    prompt = prompt + f'\n{reaction}\n'
    lines = reaction.strip().split('\n')
    line = lines[-1]
    if line.startswith("Action: Finish["):
        answer = line[15:-1]
        print(f'Answer: {answer}')
        done = True
    elif line.startswith("Action: Search["):
        query = line[15:-1]
        print(f'Will search wikipedia for "{query}"')
        document = wikipedia_search(query)
        text = document.first_chunk()
        prompt = prompt + f'\nObservation: {text}\n'
    elif line.startswith("Action: Lookup["):
        if document is None:
            print("No document defined, cannot lookup")
            done = True
        query = line[15:-1]
        print(f'Will lookup paragraph containing "{query}"')
        document = wikipedia_search(query)
        text = document.lookup(query)
        prompt = prompt + f'\nObservation: {text}\n'
    if iter >= MAX_ITER:
        print("Max iterations reached, exiting.")
        done = True
    iter += 1



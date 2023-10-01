import requests
import json
from string import Template


#question = input("Enter your question: ")
question = "What was the first major battle in the Ukrainian War?"

# load the api key from a file
with open("config.json", "r") as f:
    config = json.load(f)
    api_key = config["api_key"]

# Define a function to interact with OpenAI API
def openai_query(prompt, api_key):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "prompt": prompt,
        #"max_tokens": 350
    }
    response = requests.post("https://api.openai.com/v1/engines/davinci/completions", headers=headers, json=data)
    response_json = response.json()
    print(response_json)
    return response_json["choices"][0]["text"].strip()

# Get the Google search term
g_query = openai_query(f"When answering the following question be concise - reply with only the text of the search term, don't repeat the question. What would be a good google search term to find out the answer to the question \"{question}\"?", api_key)

if g_query[0] == '"' and g_query[-1] == '"':
    g_query = g_query[1:-1]

print(f"Testing the following query in Google search: \n{g_query}\n")

base_url = "https://www.googleapis.com/customsearch/v1"

params = {
  "key": config["google_key"],
  "cx": config["google_cx"],
  "q": g_query
}
response = requests.get(base_url, params=params)
results = response.json()

filtered = []
for result in results["items"]:
  item = {}
  item["link"] = result["link"]
  item["title"] = result["title"]
  item["snippet"] = result["snippet"]
  # Append the item to the items list
  filtered.append(item)

prompt_template = Template("""Below I give you a json formatted list of web pages. Each web page record contains the link to it, its title and a short snippet of text from it.
Please go through that list and guess if the webpages they link to can contain the answer to the question: "$question"
Please tell me which one of these links is the most promising. Don't try to answer the question itself - only judge which webpage should contain the answer.
Please answer with only a number - the index to the list of web pages - without any additional text.
Here is the list of webpages to check:
$links
""")

# format the filtered list of web pages into a json string
links = json.dumps(filtered, indent=2)

prompt = prompt_template.substitute(question=question, links=links)

print(prompt)
chosen_web_page = openai_query(prompt, api_key)

print(chosen_web_page)

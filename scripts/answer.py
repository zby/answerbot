import logging
import httpx

from openai import OpenAI
from pprint import pformat, pprint
from dotenv import dotenv_values
#from answerbot.formatter import format_markdown

from answerbot.react import LLMReactor

from answerbot.tools.wiki_tool import WikipediaTool
from answerbot.replay_client import LLMReplayClient

# Configure basic logging
logging.basicConfig(level=logging.INFO)

# Get a logger for the current module
logger = logging.getLogger(__name__)

config = dotenv_values(".env")

#from groq import Groq
#client = Groq()

#client = ReplayClient('data/conversation.json')

client = OpenAI(
     api_key=config['OPENAI_API_KEY'],
     base_url="https://oai.hconeai.com/v1",
     default_headers={
         "Helicone-Auth": f"Bearer {config['HELICONE_API_KEY']}",
     }
)
#client = OpenAI(
#     api_key=config['OPENAI_API_KEY'],
#     base_url="http://0.0.0.0:4000",
#)

completion = client.chat.completions.create(
    model='gpt-3.5-turbo',
    messages=[
        {'role': 'user', 'content': 'Hi there!'},
    ]
)

def sys_prompt(max_llm_calls):
    return f"""
Please answer the following question. You can use wikipedia for reference - but think carefully about what pages exist at wikipedia.
You have only {max_llm_calls - 1} calls to the wikipedia API.
After the first call to wikipedia you need to always reflect on the data retrieved in the previous call.
To retrieve the first document you need to call search.

When you need to know a property of something or someone - search for that something page instead of using that property in the search.
The search function automatically retrieves the first search result you don't need to call get for it.

The wikipedia pages are formatted in Markdown.
When you know the answer call finish. Please make the answer as short as possible. If it can be answered with yes or no that is best.
Remove all explanations from the answer and put them into the reasoning field.
Always try to answer the question even if it is ambiguous, just note the necessary assumptions.
"""


#client = OpenAI(
#    api_key=config['OPENAI_API_KEY'],
#    timeout=httpx.Timeout(70.0, read=60.0, write=20.0, connect=6.0)
#)

if __name__ == "__main__":

    # question = "What was the first major battle in the Ukrainian War?"
    # question = "What were the main publications by the Nobel Prize winner in economics in 2023?"
    # question = "What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?"
    # question = 'Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?'
    # question = "how old was Donald Tusk when he died?"
    # question = "how many keys does a US-ANSI keyboard have on it?"
    # question = "How many children does Donald Tusk have?"
    # question = "The director of the romantic comedy \"Big Stone Gap\" is based in what New York city?"
    #question = "When Poland became elective monarchy?"
    #question = "Were Scott Derrickson and Ed Wood of the same nationality?"
    #question = "What science fantasy young adult series, told in first person, has a set of companion books narrating the stories of enslaved worlds and alien species?"
    # question = "What year did Guns N Roses perform a promo for a movie starring Arnold Schwarzenegger as a former New York Police detective?"
    #question = "What is the weight proportion of oxygen in water?"
    #question = "Czy dane kardy kredytowej są danymi osobowymi w Polsce"
    #question = "How much is two plus two"
    #question = "Who is older, Annie Morton or Terry Richardson?"

    #question = "What are the concrete steps proposed to ensure AI safety?"
    question = 'What are the steps required to authorize the training of generative AI?'

    #question = "What is the name of the fight song of the university whose main campus is in Lawrence, Kansas and whose branch campuses are in the Kansas City metropolitan area?"
    #question = "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?"
    question = "The arena where the Lewiston Maineiacs played their home games can seat how many people?"
    question = "What is the seating capacity of Androscoggin Bank Colisée?"
    question = "Who portrayed Corliss Archer in the film Kiss and Tell?"


    max_llm_calls = 7

    reactor = LLMReactor(
        model='gpt-3.5-turbo',
        toolbox=[WikipediaTool(chunk_size=400)],
        max_llm_calls=max_llm_calls,
        client=client,
        get_system_prompt=sys_prompt,
        question_checks=[]
    )
    trace = reactor.process(question)
    print(trace.generate_report())
    print()
    print(str(trace.what_have_we_learned))
    print()
    pprint(trace.soft_errors)
    with open('data/trace.py', 'w') as file:
        file.write(repr(trace))
    with open('data/what_have_we_learned.py', 'w') as file:
        file.write(repr(trace.what_have_we_learned))
#    print(format_markdown(reactor.conversation))

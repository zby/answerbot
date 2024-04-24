import logging
import httpx
import openai

from pprint import pformat, pprint
from dotenv import load_dotenv
#from answerbot.formatter import format_markdown

from answerbot.react import get_answer_wiki

from answerbot.wikipedia_tool import WikipediaSearch
from answerbot.aae_tool import AAESearch

# Configure basic logging
logging.basicConfig(level=logging.INFO)

# Get a logger for the current module
logger = logging.getLogger(__name__)

# load OpenAI api key
load_dotenv()

#from groq import Groq
#client = Groq()

client = openai.OpenAI(timeout=httpx.Timeout(70.0, read=60.0, write=20.0, connect=6.0))

if __name__ == "__main__":

    # question = "What was the first major battle in the Ukrainian War?"
    # question = "What were the main publications by the Nobel Prize winner in economics in 2023?"
    # question = "What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?"
    # question = 'Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?'
    # question = "how old was Donald Tusk when he died?"
    # question = "how many keys does a US-ANSI keyboard have on it?"
    # question = "How many children does Donald Tusk have?"
    #question = "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?"
    # question = "The director of the romantic comedy \"Big Stone Gap\" is based in what New York city?"
    #question = "When Poland became elective monarchy?"
    question = "Were Scott Derrickson and Ed Wood of the same nationality?"
    #question = "What science fantasy young adult series, told in first person, has a set of companion books narrating the stories of enslaved worlds and alien species?"
    #question = "The arena where the Lewiston Maineiacs played their home games can seat how many people?"
    #question = "What is the name of the fight song of the university whose main campus is in Lawrence, Kansas and whose branch campuses are in the Kansas City metropolitan area?"
    # question = "What year did Guns N Roses perform a promo for a movie starring Arnold Schwarzenegger as a former New York Police detective?"
    #question = "What is the weight proportion of oxygen in water?"
    #question = "Czy dane kardy kredytowej są danymi osobowymi w Polsce"
    #question = "How much is two plus two"
    #question = "Who is older, Annie Morton or Terry Richardson?"

    #question = "What are the concrete steps proposed to ensure AI safety?"
    question = 'What are the steps required to authorize the training of generative AI?'


    config = {
        "chunk_size": 400,
        #"prompt_class": 'NERP',
        "prompt_class": 'AAE',
        "max_llm_calls": 4,
        "model": "gpt-3.5-turbo-0613",
        #"model": "gpt-4-1106-preview",
        #"model": "llama3-8b-8192",
        #'model': "mixtral-8x7b-32768",
        "question_check": 'category_and_amb',
        'reflection': 'ShortReflectionDetached',
        #'tool': WikipediaSearch,
        'tool': AAESearch,
    }

    reactor = get_answer_wiki(question, config, client)
    print(pformat(reactor.conversation))
    pprint(reactor.soft_errors)
#    print(format_markdown(reactor.conversation))

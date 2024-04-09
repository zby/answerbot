import logging

from pprint import pformat, pprint
from dotenv import load_dotenv

from answerbot.prompt_builder import System
from answerbot.prompt_templates import NoExamplesReactPrompt, Reflection, ShortReflection, QUESTION_CHECKS
from answerbot.react import get_answer_aae

# Configure basic logging
logging.basicConfig(level=logging.INFO)

# Get a logger for the current module
logger = logging.getLogger(__name__)

# load OpenAI api key
load_dotenv()

if __name__ == "__main__":

    question = "What are the concrete steps proposed to ensure AI safety?"
    question = 'What are the steps required to authorize the training of generative AI?'

    config = {
        "chunk_size": 400,
        "prompt_class": 'AAE',
        "reflection": 'None',
        "max_llm_calls": 8,
        "model": "gpt-3.5-turbo-0613",
        #"model": "gpt-4-1106-preview",
        "question_check": 'category_and_amb',
    }

    reactor = get_answer_aae(question, config)
    print(pformat(reactor.prompt.to_messages()))
    pprint(reactor.reflection_errors)




import openai
import json
import time
import logging

from .prompt_builder import FunctionalPrompt, Assistant, System, FunctionCall, FunctionResult
from .get_wikipedia import WikipediaApi

from .react_prompt import FunctionalReactPrompt, NewFunctionalReactPrompt, TextReactPrompt, NoExamplesReactPrompt
from .toolbox import WikipediaSearch
# Configure basic logging
logging.basicConfig(level=logging.INFO)

# Get a logger for the current module
logger = logging.getLogger(__name__)


class LLMReactor:
    def __init__(self, model, toolbox, prompt, summarize_prompt, max_llm_calls):
        self.model = model
        self.toolbox = toolbox
        self.prompt = prompt
        self.summarize_prompt = summarize_prompt
        self.max_llm_calls = max_llm_calls
        self.step = 0
        self.finished = False
        self.answer = None

    @staticmethod
    def convert_to_dict(obj):
        if isinstance(obj, openai.openai_object.OpenAIObject):
            return {k: LLMReactor.convert_to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [LLMReactor.convert_to_dict(item) for item in obj]
        else:
            return obj

    def openai_query(self, **args):
        if isinstance(self.prompt, FunctionalPrompt):
            # todo - we might optimize and only send the functions that are relevant
            # in particular not send the functions if function_call = 'none'
            args["functions"] = self.toolbox.functions
            if not "function_call" in args:
                args["function_call"] = "auto"
        else:
            args["stop"] = ["\nObservation:"]

        errors = []
        response = None
        for i in range(2):
            try:
                openai.api_requestor.TIMEOUT_SECS = i * 20 + 20
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=self.prompt.to_messages(),
                    **args
                )
                break
            except openai.error.Timeout as e:
                # todo logger
                print("OpenAI Timeout: ", e)
                time.sleep(20)
                errors.append(e)
                continue
            except openai.error.APIError as e:
                # todo logger
                print("OpenAI APIError: ", e)
                time.sleep(20)
                errors.append(e)
                continue
        if response is None:
            errors_string = "\n".join([str(e) for e in errors])
            raise Exception(f"OpenAI API calls failed: {errors_string}")
        response_message = response["choices"][0]["message"]
        return self.convert_to_dict(response_message)

    def set_finished(self):
        self.finished = True
    def process_prompt(self):
        logger.debug(f"Processing prompt: {self.prompt}")
        self.step += 1
        if self.step == self.max_llm_calls:
            response = self.openai_query(function_call={'name': 'finish'})
        else:
            response = self.openai_query()
        function_call = self.prompt.function_call_from_response(response)
        if function_call:
            message = FunctionCall(function_call["name"], **json.loads(function_call["arguments"]))
        else:
            message = Assistant(response.get("content"))
        logger.info(str(message))
        self.prompt.push(message)

        if function_call:
            function_args = json.loads(function_call["arguments"])
            tool_name = function_call["name"]
            if tool_name == "finish":
                answer = function_args["answer"]
                self.set_finished()
                if answer.lower() == 'yes' or answer.lower() == 'no':
                    answer = answer.lower()
                self.answer = answer
                return
            elif self.step == self.max_llm_calls:
                self.set_finished()
                logger.info("<<< Max LLM calls reached without finishing")
                return
            else:
                result = self.toolbox.process(tool_name, function_args)
                message = FunctionResult(tool_name, result)
                logger.info(str(message))
#                if len(message.content) > 500:
#                    message.summarized_below = True
                self.prompt.push(message)

                # now reflect on the observations
                message = self.summarize_prompt
                logger.info(str(message))
                self.prompt.push(message)
                response = self.openai_query(function_call='none')
                message = Assistant(response.get("content"))
                logger.info(str(message))
                self.prompt.push(message)

def get_answer(question, config):
    if 'summarize_prompt' not in config:
        #config['summarize_prompt'] = System("Please extract the relevant facts from the data above, note which sections of the current page could contain more relevant information and plan next steps.")
        config['summarize_prompt'] = System("Reflect on the received information and plan next steps.")
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
        'TRP': TextReactPrompt,
        'NERP': NoExamplesReactPrompt,
    }
    prompt_class = CLASS_MAP[config['prompt']]
    prompt = prompt_class(question, config['example_chunk_size'])
    wiki_api = WikipediaApi(max_retries=2, chunk_size=config['chunk_size'])
    toolbox = WikipediaSearch(wiki_api)
    reactor = LLMReactor(config['model'], toolbox, prompt, config['summarize_prompt'], config['max_llm_calls'])
    while True:
        print()
        print(f">>>LLM call number: {reactor.step}")
        answer = reactor.process_prompt()
        #print(prompt.parts[-2])
        if reactor.finished:
            return reactor
#        if 'gpt-4' in config['model']:
#            time.sleep(59)


from dataclasses import dataclass
from functools import wraps
from time import sleep, time
from typing import Annotated, Callable, TypeVar
from llm_easy_tools.tool_box import ToolBox, llm_function
from openai import OpenAI
import httpx
from openai.types.chat import ChatCompletionNamedToolChoiceParam, ChatCompletionToolChoiceOptionParam
from openai.types.chat.completion_create_params import Function
import logging

from answerbot.replay_client import ReplayClient

import json

DEFAULT_SYSTEM_PROMPT = '''
You are an expert system for answering questions in a specific domain, specifically {domain}.
You are to answer user's questions in a comprehensive manner. You can access various python 
functions, but do consider the costs. Every function that has a cost will have

COST: <number>

in it's description. A cost is a number which will be reduced from your ENERGY pool. After you've
depleted your energy pool, you will no longer be able to call these costly functions.

You will initially have the energy pool of {initial_energy}.

USE COSTLY FUNCTIONS CAREFULLY, but try to deplete all your available energy to provide the
MOST COMPREHENSIVE ANSWER POSSIBLE.

Functions whose names end with _get_paragraphs will return a document as a list of paragraphs,
like

number: first few words
number: first few words

You can then use `read_paragraphs` function, providing indices of the paragraphs you wish to read.
Keep in mind that it only works with the last opened document. This function is very expensive, so
use it carefully.

If you've read the paragraphs once and think you need more, you can use `read_paragraphs` again.

Once you've read a paragraph that you believe is relevant to user's question, you must call
`reflect` function. DO THIS EVERY TIME YOU FOUND A RELEVANT PARAGRAPH, BUT ONLY IF IT IS RELEVANT.

When you are ready to provide an answer, call `finish` function. Only call `finish` if you are
ready to provide PRECISE and COMPREHENSIVE answer. DO NOT call `finish` if there are any additional
steps to be taken and you still have energy to do so.
'''


@dataclass(frozen=True)
class Answer:
    answer: str
    reasoning: str


def cost(x: int):
    def _wrapper(f):
        @wraps(f)
        def _w(self: 'LLMReactor', *args, **kwargs):
            self.energy -= x
            return f(self, *args, **kwargs)
        _w.__doc__ = getattr(_w, '__doc__', '') + f'\nCOST: {x}'
        setattr(_w, '__cost__', x)
        return f
    return _wrapper


class LLMReactor:
    def __init__(self, 
                 system_prompt: str,
                 model: str,
                 client: OpenAI,
                 question: str,
                 toolbox: ToolBox,
                 energy: int=100,
                 throttle: float=10.,
                 followup_assumptions: list[str] = []
                 ):
        self.energy = energy
        self.question = question
        self._model = model
        self._client = client or OpenAI(
                timeout=httpx.Timeout(70, read=60.0, write=20.0, connect=6.0)
                )
        self.answer = None
        self._last_query = 0
        self._throttle=throttle
        self._messages = []
        self._toolbox = toolbox
        self._toolbox.register_function(self.finish)

    def __call__(self):
        self._add_system_message()
        self._add_user_question()
        assumptions = self._add_assumptions()
        self.before_loop()

        while self.answer is None:
            self._step()

        # write all messages to json file
        with open('data/messages.json', 'w') as f:
            json.dump(self._messages, f, indent=4)

    def before_loop(self):
        pass

    def _step(self):
        self.throttle()

        if self.energy > 0:

            response = self._client.chat.completions.create(
                    model=self._model,   
                    messages=self._messages,
                    tools=self._toolbox.tool_schemas(predicate=lambda x: getattr(x, '__cost__', 0) <= max(0, self.energy))
                    )
        else:
            response = self._client.chat.completions.create(
                    model=self._model,   
                    messages=self._messages,
                    tools=[self._toolbox.get_tool_schema('finish')], 
                    tool_choice=ChatCompletionNamedToolChoiceParam(
                        type='function',
                        function=Function(name='finish'),
                        )
                    )


        self._last_query = time()
        content = response.choices[0].message.content
        self._messages.append(response.choices[0].message.model_dump())
        print(response)
        tool_results = self._toolbox.process_response(response)
        self._messages.extend(x.to_message() for x in tool_results)

        if content:
            self._followup_assumptions.append(content)

        assistant_message = (
                f'you have {self.energy} left to spend' if self.energy > 0 else
                'You have no more energy left to spend. Give the answer using the information you already have'
                )
        assistant_message += (
                '\nProvide a summary of the information gathered so far, and reflect on\n'
                "\nwhat are the next steps required to provide the precise answer to the "
                "user's question\n "
                )

        self._messages.append({'role': 'assistant', 'content': assistant_message})

    def _add_system_message(self):
        self._messages.append({
            'role': 'system',
            'content': self.system_prompt.format(
                domain=self.DOMAIN,
                initial_energy=self.energy,
                )
            })

    def _add_user_question(self):
        self._messages.append({
            'role': 'user',
            'content': self.question,
            })

    def _add_assumptions(self) -> str|None:
        self._messages.append({
            'role': 'assistant',
            'content': 'What are the assumptions that can be made about the question?'
            })
        response = self._client.chat.completions.create(
                model=self._model,
                messages=self._messages
                )
        result = response.choices[0].message.content
        self._messages.append(response.choices[0].message.model_dump())
        return result


    def finish(
            self,
            answer: Annotated[str, "The answer to user's original question"],
            reasoning: Annotated[str, 'Your reasoning for the answer']
            ):
        self.answer = Answer(answer=answer, reasoning=reasoning)
        return 'finished'

    def throttle(self):
        # if client is ReplayClient, do nothing
        if isinstance(self._client, ReplayClient):
            return
        wait = max(0, self._last_query + self._throttle - time())
        logging.getLogger(__name__).info(f'Waiting {wait} seconds')
        sleep(wait)
        self._last_query = time()



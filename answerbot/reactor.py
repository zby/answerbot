from dataclasses import dataclass
from functools import wraps
import inspect
from time import time
from typing import Annotated, Callable
from llm_easy_tools.tool_box import ToolBox, llm_function
from openai import OpenAI
import httpx


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


@dataclass(frozen=True)
class RelevantParagraph:
    document: str
    paragraph: str


@dataclass(frozen=True)
class ReactorResponse:
    question: str
    answer: Answer
    relevant_paragraphs: list[RelevantParagraph]
    assumptions: str|None
    followup_assumptions: list[str]


def paragraphs(f: Callable[..., tuple[str, list[str]]]):
    @wraps(f)
    def _w(self, *args, **kwargs):
        result = f(self, *args, **kwargs)
        self._document, self._paragraphs = result
        return 'Paragraphs in the current document:\n' + '\n'.join(
                f'{i}: {s[:self._paragraph_size]}' for i, s in enumerate(self._paragraphs)
                )
    return _w



def cost(x: int):
    def _wrapper(f):
        f.__doc__ += f'\nCOST: {x}'
        f.__cost__ = x
        return f
    return _wrapper


class LLMReactor:
    SYSTEM_PROMPT: str = DEFAULT_SYSTEM_PROMPT
    DOMAIN: str = 'Undefined domain'
    READ_PARAGRAPHS_COST = 10

    def __init__(self, 
                 model: str,
                 client: OpenAI,
                 question: str,
                 energy: int=100,
                 throttle: float=10.,
                 paragraph_size: int=55):
        self.energy = energy
        self.question = question
        self._model = model
        self._client = client or OpenAI(
                timeout=httpx.Timeout(70, read=60.0, write=20.0, connect=6.0)
                )
        self.answer = None
        self.relevant_paragraphs: list[RelevantParagraph] = []
        self._document: str|None = None
        self._paragraphs: list[str]|None = None
        self._paragraph_size=paragraph_size
        self._followup_assumptions = []
        self._last_query = 0
        self._throttle=throttle
        self._messages = []
        self._toolbox = ToolBox()
        self._toolbox.register_toolset(self)

    def __call__(self) -> ReactorResponse:
        self._add_system_message()
        self._add_user_question()
        assumptions = self._add_assumptions()
        self.before_loop()

        while self.answer is None:
            self._step()

        return ReactorResponse(
                question=self.question,
                answer=self.answer,
                relevant_paragraphs=self.relevant_paragraphs,
                assumptions=assumptions,
                followup_assumptions=self._followup_assumptions,
                )

    def before_loop(self):
        while self._last_query+self._throttle > time():
            continue
        self._last_query = time()
        pass

    def _step(self):
        while self._last_query+self._throttle > time():
            continue
        self._last_query = time()

        response = self._client.chat.completions.create(
                model=self._model,   
                messages=self._messages,
                tools=self._toolbox.tool_schemas(predicate=lambda x: getattr(x, '__cost__', 0) <= max(0, self.energy))
                )
        content = response.choices[0].message.content
        self._messages.append(response.choices[0].message)
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
            'content': self.SYSTEM_PROMPT.format(
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
        self._messages.append(response.choices[0].message)
        return result
        

    def _get_tools(self) -> list[Callable]:
        return [
                method
                for _, method in inspect.getmembers(self)
                if callable(method)
                and getattr(method, '__llm_function__', False)
                and self.energy > getattr(method, '__cost__', 0)
                ]

    @llm_function()
    @cost(10)
    def read_paragraphs(
            self,
            paragraphs: Annotated[list[int], 'The indices of paragraphs to read']
            ):
        ''' Return the full text of paragraphs at specified indices '''
        if not self._paragraphs:
            raise RuntimeError('No document is currently open')
        result = ''
        for index in paragraphs:
            try:
                p = self._paragraphs[index]
            except IndexError:
                continue
            result += f'# Paragraph {index}:\n'
            result += p + '\n'
        self.energy -= 10
        return result

        
    @llm_function()
    def reflect(
            self,
            paragraphs: Annotated[list[int], 'The indices of paragraphs you found to be relevant AND did contain the required information']
            ):
        assert self._paragraphs is not None
        assert self._document is not None
        for index in paragraphs:
            try:
                self.relevant_paragraphs.append(
                        RelevantParagraph(
                            document=self._document,
                            paragraph=self._paragraphs[index]
                            )
                        )
            except IndexError:
                continue
        return str(paragraphs)


    @llm_function()
    def finish(
            self,
            answer: Annotated[str, "The answer to user's original question"],
            reasoning: Annotated[str, 'Your reasoning for the answer']
            ):
        self.answer = Answer(answer=answer, reasoning=reasoning)
        return 'finished'


import time
import logging
import copy
from typing import Literal, Union, List, Dict, Annotated, Optional, Callable, Any, Protocol, Iterable, runtime_checkable
from pprint import pprint
from dataclasses import dataclass, field
import traceback

from litellm import completion
import litellm

from .prompt_templates import QUESTION_CHECKS, PROMPTS, REFLECTIONS 

from llm_easy_tools import process_response, get_tool_defs, ToolResult
from answerbot.tools.wiki_tool import Observation, InfoPiece
from answerbot.clean_reflection import ReflectionResult, KnowledgeBase
from answerbot.trace import Trace, Question

# Configure basic logging
logging.basicConfig(level=logging.ERROR)

# Get a logger for the current module
logger = logging.getLogger(__name__)

# Global configuration
litellm.num_retries = 3
litellm.retry_delay = 10
litellm.retry_multiplier = 2

@runtime_checkable
class HasLLMTools(Protocol):
    def get_llm_tools(self) -> Iterable[Callable]:
        pass

@dataclass(frozen=True)
class LLMReactor:
    model: str
    toolbox: list[Callable|HasLLMTools]
    max_llm_calls: int
    system_prompt: str
    question_checks: list[str] = field(default_factory=list)

    class LLMReactorError(Exception):
        pass

    def delegate(self, question: str) -> str:
        """
        Delegate the question to an expert.
        """
        # The function docstring is probably not enough information to delegate the question to an expert
        # So it needs to be overriden by wrapping delegate in LLMFunction

        print(f'Delegating question: "{question}" to an expert')

        trace = self.process(question)
        return trace.answer


    def process(self, question: str) -> Trace:
        trace = Trace()
        trace.append({'role': 'system', 'content': self.system_prompt})
        trace.append(Question(question, self.max_llm_calls))

        while trace.answer is None and trace.step <= self.max_llm_calls + 1:
            self.one_step(trace)
            trace.step += 1
            if trace.step == self.max_llm_calls:
                step_info = "\n\nThis was the last data you can get - in the next step you need to formulate your answer"
            else:
                step_info = f"\n\nThis was {trace.step} out of {self.max_llm_calls} calls for data."
            trace.append({'role': 'user', 'content': step_info})

        return trace

    def one_step(self, trace: Trace) -> None:
        tools = self.get_tools(trace)
        schemas = get_tool_defs(tools)
        response = self.openai_query(trace, schemas)
        results = process_response(response, tools)
        result = results[0]
        if result.error is not None:
            print(result.error)
            raise self.LLMReactorError(result.stack_trace)
        trace.append(result)
        for result in results:
            if isinstance(result.output, Observation) and result.output.reflection_needed():
                self.process_reflection(result.output, trace)

    def process_reflection(self, observation: Observation, trace: Trace) -> None:
        reflection_trace = self.create_reflection_trace(observation, trace)
        reflection = self.analyze_retrieval(reflection_trace)
        reflection_string = reflection_trace.update_knowledge_base(reflection, observation)
        print(reflection_string)
        trace.append(reflection_trace)

        planning_trace = self.plan_next_action(reflection_string, observation, trace)
        trace.append(planning_trace)

    def create_reflection_trace(self, observation: Observation, trace: Trace) -> Trace:
        new_trace = Trace()
        sysprompt = "You are a researcher working on a user question in a team with other researchers. You need to check the assumptions that other researchers made."
        learned_stuff = f"\n\nSo far we have some notes on the following urls:{trace.what_have_we_learned.learned()}" if not trace.what_have_we_learned.is_empty() else ""
        user_prompt = f"""# Question

The user's question is: {trace.user_question()}{learned_stuff}

# Retrieval

We have performed information retrieval with the following results:
{str(observation)}

# Current task

You need to review the information retrieval recorded above and reflect on it.
You need to note all information that can help in answering the user question that can be learned from the retrieved fragment,
together with the quotes that support them.
Please remember that the retrieved content is only a fragment of the whole page.
"""

        new_trace.append({'role': 'system', 'content': sysprompt})
        new_trace.append({'role': 'user', 'content': user_prompt})
        return new_trace

    def analyze_retrieval(self, reflection_trace: Trace) -> ReflectionResult:
        schemas = get_tool_defs([ReflectionResult])
        response = self.openai_query(reflection_trace, schemas)
        results = process_response(response, [ReflectionResult])
        new_result = results[0]
        reflection_trace.append(new_result)
        if new_result.error is not None:
            raise self.LLMReactorError(new_result.stack_trace)
        return new_result.output

    def plan_next_action(self, reflection: ReflectionResult, observation: Observation, trace: Trace) -> Trace:
        planning_trace = Trace()
        sysprompt = "You are a researcher working on a user question in a team with other researchers. You need to check the assumptions that the other researchers made."
        user_prompt = f"""# Question

The user's question is: {trace.user_question()}

# Available tools

{observation.available_tools}

# Retrieval

We have performed information retrieval with the following results:

{observation}

# Reflection

{reflection}

# Next step

What would you do next?
Please analyze the retrieved data and check if you have enough information to answer the user question.

If you still need more information, consider the available tools.

You need to decide if the current page is relevant to answer the user question.
If it is, then you should recommed exploring it further with the `lookup` or `read_more` tools.

When using `search` please use simple queries. When trying to learn about a property of an object or a person,
first search for that object then you can browse the page to learn about its properties.
For example to learn about the nationality of a person, first search for that person.
If the persons page is retrieved but the information about nationality is not at the top of the page
you can use `read_more` to continue reading or call `lookup('nationality')` or `lookup('born')` to get more information.

Please specify both the tool and the parameters you need to use if applicable.
Explain your reasoning."""

        planning_trace.append({'role': 'system', 'content': sysprompt})
        planning_trace.append({'role': 'user', 'content': user_prompt})

        response = self.openai_query(planning_trace)
        planning_result = response.choices[0].message.content

        reflection_string = f"**My Notes**\n{reflection}\n\nHmm what I could do next?\n\n{planning_result}"
        planning_trace.result = {'role': 'user', 'content': reflection_string}
        return planning_trace


    def get_tools(self, trace: Trace) -> list[Callable]:
        tools = [trace.finish]
        if trace.step < self.max_llm_calls + 1:
            for item in self.toolbox:
                if isinstance(item, HasLLMTools):
                    new_tools = item.get_llm_tools()
                    tools.extend(new_tools)
                else:
                    tools.append(item)
        return tools

    def openai_query(self, trace, schemas=[]):
        messages = trace.to_messages()
        args = {
            'model': self.model,
            'messages': messages
        }

        if len(schemas) > 0:
            args['tools'] = schemas
            if len(schemas) == 1:
                args['tool_choice'] = {'type': 'function', 'function': {'name': schemas[0]['function']['name']}}
            else:
                args['tool_choice'] = "auto"

        result = completion(**args)
        message = result.choices[0].message

        # Remove all but one tool_call from the result if present
        # With many tool calls the LLM gets confused about the state of the tools
        if hasattr(message, 'tool_calls') and message.tool_calls:
            #print(f"Tool calls: {result.choices[0].message.tool_calls}")
            if len(message.tool_calls) > 1:
                message.tool_calls = [message.tool_calls[0]]
                trace.soft_errors.append(f"More than one tool call: {message.tool_calls}")

        if len(schemas) > 0:
            if not hasattr(message, 'tool_calls') or not message.tool_calls:
                stack_trace = traceback.format_stack()
                trace.soft_errors.append(f"No function call:\n{stack_trace}")

        trace.append(message)

        return result

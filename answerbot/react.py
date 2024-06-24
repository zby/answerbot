import time
import logging
import copy
from typing import Literal, Union, List, Dict, Annotated, Optional, Callable, Any, Protocol, Iterable, runtime_checkable
from pprint import pprint
from dataclasses import dataclass, field
import traceback

from litellm import completion

from .prompt_templates import QUESTION_CHECKS, PROMPTS, REFLECTIONS 

from llm_easy_tools import process_response, get_tool_defs, ToolResult
from answerbot.tools.wiki_tool import Observation, InfoPiece
from answerbot.clean_reflection import ReflectionResult, KnowledgeBase
from answerbot.trace import Trace, Question

# Configure basic logging
logging.basicConfig(level=logging.ERROR)

# Get a logger for the current module
logger = logging.getLogger(__name__)

@runtime_checkable
class HasLLMTools(Protocol):
    def get_llm_tools(self) -> Iterable[Callable]:
        pass

@dataclass(frozen=True)
class LLMReactor:
    model: str
    toolbox: list[Callable|HasLLMTools]
    max_llm_calls: int
    get_system_prompt: Callable[[str], str]
    question_checks: list[str] = field(default_factory=list)

    class LLMReactorError(Exception):
        pass

    def process(self, question: str) -> Trace:
        trace = Trace()
        trace.append({'role': 'system', 'content': self.get_system_prompt(self.max_llm_calls)})
        trace.append(Question(question))

        self.analyze_question(trace)
        while trace.answer is None and trace.step <= self.max_llm_calls + 1:
            self.one_step(trace)
            trace.step += 1
            if trace.step == self.max_llm_calls:
                step_info = "\n\nThis was the last data you can get - in the next step you need to formulate your answer"
            else:
                step_info = f"\n\nThis was {trace.step} out of {self.max_llm_calls} calls for data."
            trace.append({'role': 'system', 'content': step_info})

        return trace

    def one_step(self, trace: Trace) -> None:
        tools = self.get_tools(trace)
        schemas = get_tool_defs(tools)
        response = self.openai_query(trace.to_messages(), schemas)
        message = response.choices[0].message
        if len(schemas) > 0:
            if not message.tool_calls:
                stack_trace = traceback.format_stack()
                trace.soft_errors.append(f"No function call:\n{stack_trace}")

        trace.append(message)
        results = process_response(response, tools)
        for result in results:
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
        learned_stuff = f"\n\nSo far we have some notes on the following urls:{trace.what_have_we_learned.learned()}" if not trace.what_have_we_learned.is_empty() else ""
        sysprompt = f"""You are a researcher working on the following user question:
{trace.user_question()}{learned_stuff}

You need to review the information retrieval recorded below."""

        user_prompt = str(observation)
        new_trace.append({'role': 'system', 'content': sysprompt})
        new_trace.append({'role': 'user', 'content': user_prompt})
        return new_trace

    def analyze_retrieval(self, reflection_trace: Trace) -> ReflectionResult:
        schemas = get_tool_defs([ReflectionResult])
        response = self.openai_query(reflection_trace.to_messages(), schemas)
        message = response.choices[0].message
        reflection_trace.append(message)
        results = process_response(response, [ReflectionResult])
        if len(results) > 1:
            reflection_trace.soft_errors.append(f"More than one reflection result")
        new_result = results[0]
        reflection_trace.append(new_result)
        if new_result.error is not None:
            raise self.LLMReactorError(new_result.stack_trace)
        return new_result.output

    def plan_next_action(self, reflection: ReflectionResult, observation: Observation, trace: Trace) -> Trace:
        planning_trace = Trace()
        sysprompt = f"""You are a researcher working on the following user question:
{trace.user_question()}

You need to plan the next action based on the reflection below."""

        user_prompt = f"""Reflection:
{reflection}

What would you do next?
Please analyze the retrieved data and check if you have enough information to answer the user question. Explain your reasoning.

If you still need more information, consider the available tools:
{self.format_available_tools(observation.available_tools)}
"""

        planning_trace.append({'role': 'system', 'content': sysprompt})
        planning_trace.append({'role': 'user', 'content': user_prompt})

        response = self.openai_query(planning_trace.to_messages(), [])
        planning_result = response.choices[0].message.content

        planning_trace.append({'role': 'assistant', 'content': planning_result})
        return planning_trace

    def format_available_tools(self, tools: list[str]) -> str:
        return "\n".join([f"  - {tool}" for tool in tools])

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

    def openai_query(self, messages, schemas=[]):
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

        return result

    def analyze_question(self, trace: Trace):
        for query in self.question_checks:
            question_check = { 'role': 'user', 'content': query }
            logger.info(str(question_check))
            trace.append(question_check)
            response = self.openai_query(trace.to_messages())
            message = response.choices[0].message
            trace.append(message)
            message = { 'role': 'user', 'content': 'OK' }  # This is because Anthropic cannot handle function calls after an "assistant" message
            trace.append(message)

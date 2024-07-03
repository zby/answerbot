import logging
import copy
from typing import Literal, Union, Callable, Any, Protocol, Iterable, runtime_checkable, Optional
from pprint import pprint
from dataclasses import dataclass, field

from llm_easy_tools import process_response, get_tool_defs, ToolResult
from answerbot.tools.wiki_tool import Observation, InfoPiece
from answerbot.clean_reflection import ReflectionResult, KnowledgeBase
from answerbot.trace import Trace, Question
from answerbot.reflector import Reflector, Planner

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
    system_prompt: str
    user_prompt_template: str
    reflector: Optional[Reflector] = Reflector()
    planner: Optional[Planner] = Planner()

    class LLMReactorError(Exception):
        pass


    def process(self, question: str) -> Trace:
        print("Processing question:", question)
        trace = Trace()
        trace.append({'role': 'system', 'content': self.system_prompt})
        trace.append(Question(self.user_prompt_template, question, self.max_llm_calls - 1))

        what_have_we_learned = KnowledgeBase()

        while trace.answer is None and trace.step <= self.max_llm_calls + 1:
            self.one_step(trace, what_have_we_learned)
            trace.step += 1
            if trace.step == self.max_llm_calls:
                step_info = "\n\nThis was the last data you can get - in the next step you need to formulate your answer"
            else:
                step_info = f"\n\nThis was {trace.step} out of {self.max_llm_calls} calls for data."
            trace.append({'role': 'user', 'content': step_info})

        return trace

    def one_step(self, trace: Trace, what_have_we_learned: KnowledgeBase) -> None:
        tools = self.get_tools(trace)
        schemas = get_tool_defs(tools)
        response = trace.openai_query(self.model, schemas)
        results = process_response(response, tools)
        result = results[0]
        if result.error is not None:
            print(result.error)
            raise self.LLMReactorError(result.stack_trace)
        trace.append(result)
        self.reflect_and_plan(trace, result.output, what_have_we_learned)

    def reflect_and_plan(self, trace: Trace, output, what_have_we_learned: KnowledgeBase) -> None:
        if isinstance(output, Observation) and output.reflection_needed():
            observation = output
            reflection_trace = self.reflector.reflect(self.model, observation, trace.user_question(), what_have_we_learned)
            trace.append(reflection_trace)
            reflection_string = reflection_trace.hidden_result

            planning_trace = self.planner.plan_next_action(self.model, reflection_string, observation, trace.user_question())
            trace.append(planning_trace)

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


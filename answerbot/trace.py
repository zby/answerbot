from litellm.types.utils import Message
from dataclasses import dataclass, field, asdict

from typing import Union, Any, Optional, Annotated
from pprint import pprint

from llm_easy_tools import ToolResult
from llm_easy_tools.processor import ToContent
from answerbot.clean_reflection import ReflectionResult, KnowledgeBase
from answerbot.tools.observation import Observation

import traceback

from litellm import completion
import litellm

# Global configuration
litellm.num_retries = 3
litellm.retry_delay = 10
litellm.retry_multiplier = 2


@dataclass(frozen=True)
class Prompt:
    prompt_template: str

    def to_message(self) -> dict:
        # Create a dictionary of all fields, excluding prompt_template
        fields = {k: v for k, v in asdict(self).items() if k != 'prompt_template'}
        content = self.prompt_template.format(**fields)
        return {
            'role': 'user',
            'content': content.strip()
        }

@dataclass(frozen=True)
class Question(Prompt):
    question: str
    max_llm_calls: int


@dataclass
class Trace(ToContent):
    entries: list[Union[dict, Prompt, Message, ToolResult, 'Trace']] = field(default_factory=list)
    result: Optional[dict] = None
    hidden_answer: Optional[str] = None
    step: int = 0
    answer: Optional[object] = None
    soft_errors: list[str] = field(default_factory=list)
    reflection_prompt: list[str] = field(default_factory=list)

    def to_content(self):
        return self.generate_report()

    def append(self, entry):
        self.entries.append(entry)

    def to_messages(self) -> list[dict]:
        """
        Returns:
        list[dict]: A list of dictionaries representing the messages and tool results.
        """
        all_messages = []
        for entry in self.entries:
            if isinstance(entry, Trace):
                if entry.result is not None:
                    all_messages.append(entry.result)
                if entry.answer is not None:
                    all_messages.append({'role': 'assistant', 'content': entry.generate_report()})
            elif isinstance(entry, ToolResult):
                all_messages.append(entry.to_message())
            elif isinstance(entry, Message):
                all_messages.append(entry.model_dump())
            elif isinstance(entry, dict):
                all_messages.append(entry)
            elif isinstance(entry, Prompt):
                all_messages.append(entry.to_message())
            else:
                raise ValueError(f'Unsupported entry type: {type(entry)}')
        return all_messages

    def user_question(self) -> Optional[str]:
        for entry in self.entries:
            if isinstance(entry, Question):
                return entry.question
        return None

    def __str__(self):
        entries_str = ''
        for entry in self.entries:
            entry_str = str(entry)
            entry_str = entry_str.replace('\n', '\n    ')
            entries_str += f"\n    {entry_str}"
        return (f"Trace(\n"
                f"    entries=[{entries_str}],\n"
                f"    result={str(self.result)},\n"
                f"    step={self.step},\n"
                f"    answer={self.answer},\n"
                f"    soft_errors={self.soft_errors},\n"
                f"    reflection_prompt={self.reflection_prompt},\n"
                f")")

    def length(self):
        length = 0
        for entry in self.entries:
            if isinstance(entry, Trace):
                length += entry.length()
            else:
                length += 1
        return length


    def generate_report(self) -> str:
        report = f'''
The answer to the question:"{self.user_question()}" is:
{str(self.answer)}
'''
        return report

    def openai_query(self, model, schemas=[]):
        messages = self.to_messages()
        args = {
            'model': model,
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
                self.soft_errors.append(f"More than one tool call: {message.tool_calls}")
                message.tool_calls = [message.tool_calls[0]]

        if len(schemas) > 0:
            if not hasattr(message, 'tool_calls') or not message.tool_calls:
                stack_trace = traceback.format_stack()
                self.soft_errors.append(f"No function call:\n{stack_trace}")

        self.append(message)

        return result



# Example usage
if __name__ == "__main__":
    user_prompt_template = "Question: {question}\nMax LLM calls: {max_llm_calls}"

    #question = Question(user_prompt_template, question='What is the distance from Moscow to Sao Paulo?', max_llm_calls=10)
    #print(question.to_message())

    trace = Trace()
    trace.append(Question(user_prompt_template, 'What is the distance from Moscow to Sao Paulo?', 10))
    trace.append({'node1': 'value1'})
    trace.append(ToolResult(tool_call_id='tool_call_id', name='tool_name', output='tool_result'))

    sub_trace = Trace(
        [Question(user_prompt_template, 'Where is Sao Paulo?', 10)],)
    trace.append(ToolResult(tool_call_id='tool_call_id', name='process', output=sub_trace))

    # print main trace messages
    pprint(trace.to_messages())

#    pprint(sub_trace.to_messages())

    print(str(trace))

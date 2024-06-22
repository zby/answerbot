from openai.types.chat import ChatCompletionMessage
from dataclasses import dataclass, field

from typing import Union, Any, Optional, Annotated
from pprint import pprint

from llm_easy_tools import ToolResult
from answerbot.clean_reflection import ReflectionResult, KnowledgeBase
from answerbot.tools.observation import Observation

@dataclass(frozen=True)
class Question:
    question: str
    
    def to_message(self):
        return {'role': 'user', 'content': f"Question: {self.question}" }

@dataclass(frozen=True)
class Answer:
    answer: str
    answer_short: str
    reasoning: str

    def normalized_answer(self):
        answer = self.answer
        answer = answer.strip(' \n.\'"')
        answer = answer.replace('’', "'")  # Replace all curly apostrophes with straight single quotes
        answer = answer.replace('"', "'")  # Replace all double quotes with straight single quotes
        if answer.lower() == 'yes' or answer.lower() == 'no':
            answer = answer.lower()
        return answer

    def __str__(self):
        return f'{self.normalized_answer()}\n\nReasoning: {self.reasoning}'



@dataclass
class Trace:
    entries: list[Union[dict, Question, ChatCompletionMessage, ToolResult, 'Trace']] = field(default_factory=list)
    result: Optional[dict] = None
    step: int = 0
    answer: Optional[Answer] = None
    soft_errors: list[str] = field(default_factory=list)
    reflection_prompt: list[str] = field(default_factory=list)
    what_have_we_learned: KnowledgeBase = field(default_factory=KnowledgeBase)

    def append(self, entry):
        self.entries.append(entry)

    def set_answer(self, answer: str, answer_short: str, reasoning: str):
        self.answer = Answer(answer, answer_short, reasoning)

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
            elif isinstance(entry, ToolResult):
                all_messages.append(entry.to_message())
            elif isinstance(entry, ChatCompletionMessage):
                all_messages.append(entry.model_dump())
            elif isinstance(entry, dict):
                all_messages.append(entry)
            elif isinstance(entry, Question):
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
                f"    what_have_we_learned={self.what_have_we_learned}\n"
                f")")

    def length(self):
        length = 0
        for entry in self.entries:
            if isinstance(entry, Trace):
                length += entry.length()
            else:
                length += 1
        return length

    def update_knowledge_base(self, reflection: ReflectionResult, observation: Observation) -> str:
        knowledge_piece = reflection.extract_knowledge(observation)
        self.what_have_we_learned.add_knowledge_piece(knowledge_piece)
        reflection.remove_checked_urls(self.what_have_we_learned.urls())
        reflection_string = f"current url: {knowledge_piece.url}\n"
        if len(reflection.new_sources) > 0 or not knowledge_piece.is_empty():
            reflection_string += f"{str(knowledge_piece)}\n"
            if len(reflection.new_sources) > 0:
                reflection_string += f"Discovered new sources: {reflection.new_sources}"
        return reflection_string


    def generate_report(self) -> str:
        report = f'''
The answer to the question:"{self.user_question()}" is:
{str(self.answer)}
'''
        return report

    def finish(self,
               answer: Annotated[str, "The answer to the user's question"],
               answer_short: Annotated[str, "A short version of the answer"],
               reasoning: Annotated[str, "The reasoning behind the answer. Think step by step. Mention all assumptions you make."],
    ):
        """
        Finish the task and return the answer.
        """
        self.set_answer(answer, answer_short, reasoning)
        return answer



# Example usage
if __name__ == "__main__":
    trace = Trace([Question('What is the distance from Moscow to Sao Paulo?')])
    trace.append({'node1': 'value1'})
    trace.append(ToolResult(tool_call_id='tool_call_id', name='tool_name', output='tool_result'))

    sub_trace = Trace([trace.entries[0]], {'role': 'assistant', 'content': 'sub_trace_result'})
    trace.append(sub_trace)

    # print main trace messages
    pprint(trace.to_messages())

    pprint(sub_trace.to_messages())

    print(str(trace))

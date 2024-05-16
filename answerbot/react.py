import json
import time
import logging
import copy
from pydantic import BaseModel, Field, field_validator, ValidationError
from typing import Literal, Union, List, Dict, Annotated, Optional, Callable, Any
from pprint import pprint
from dataclasses import dataclass

from openai.types.chat.chat_completion import ChatCompletionMessage

from .prompt_templates import QUESTION_CHECKS, PROMPTS, REFLECTIONS 

from llm_easy_tools import process_response, get_tool_defs, get_toolset_tools, ToolResult
from answerbot.wiki_tool import Observation

# Configure basic logging
logging.basicConfig(level=logging.INFO)

# Get a logger for the current module
logger = logging.getLogger(__name__)

class Trace:
    def __init__(self, entries = None, user_question=None):
        self.entries = [] if entries is None else entries
        self.user_question = user_question

    def add_entry(self, entry):
        self.entries.append(entry)

    def add_system_message(self, content):
        self.entries.append({ 'role': 'system', 'content': content })

    def add_user_question(self, content):
        self.entries.append({ 'role': 'user', 'content': f"Question: {content}" })
        self.user_question = content

    def to_messages(self) -> List[Dict]:
        """
        Returns:
        List[Dict]: A list of dictionaries representing the messages and tool results.
        """
        all_messages = []
        for entry in self.entries:
            if isinstance(entry, dict):
                all_messages.append(entry)
            elif isinstance(entry, BaseModel):
                all_messages.append(entry.model_dump())
            elif isinstance(entry, ToolResult):
                all_messages.append(entry.to_message())
            else:
                raise ValueError(f"Invalid entry type: {type(entry)}")
        return all_messages

    def __repr__(self):
        return f"Trace(entries={self.entries}, user_question={self.user_question!r})"

    def generate_report(self):
        """
        Generates a report from a Trace object containing the user question, the answer, and the list of document quotes used.

        Args:
        trace (Trace): The Trace object containing the entries of the conversation.

        Returns:
        str: A formatted report as a string.
        """
        report = []
        answer = None
        document_quotes = []

        for entry in self.entries:
            if isinstance(entry, ToolResult):
                if isinstance(entry.output, Observation):
                    if entry.output.info_pieces:
                        for info_piece in entry.output.info_pieces:
                            if info_piece.quotable:
                                document_quotes.append(f"{info_piece.source}: {info_piece.text}\n\n---")
                elif isinstance(entry.output, Answer):
                    answer = entry.output

        report.append(f"User Question: {self.user_question}")
        report.append(f"Answer: {answer.normalized_answer()}")
        report.append(f"Reasoning: {answer.reasoning}")
        report.append("Analyzed Document Fragments:")
        report.extend(document_quotes)

        return "\n".join(report)



@dataclass
class Answer:
    answer: str
    answer_short: str
    reasoning: str
    ambiguity: Optional[str]

    def normalized_answer(self):
        answer = self.answer
        answer = answer.strip(' \n.\'"')
        answer = answer.replace('’', "'")  # Replace all curly apostrophes with straight single quotes
        answer = answer.replace('"', "'")  # Replace all double quotes with straight single quotes
        if answer.lower() == 'yes' or answer.lower() == 'no':
            answer = answer.lower()
        return answer

    def __str__(self):
        return self.normalized_answer()




class LLMReactor:
    class LLMReactorError(Exception):
        pass

    def __init__(self, 
                 model: str,
                 toolbox: list[Callable],
                 trace: Trace,
                 max_llm_calls: int,
                 client: object,
                 reflection: Any,
                 soft_reflection_validation=True,
                 question_checks=None,
                 case_insensitive=False,
                 ):
        self.model = model
        self.toolbox = toolbox
        self.toolbox.append(self.finish)
        self.trace = trace
        self.max_llm_calls = max_llm_calls
        self.client = client
        self.soft_reflection_validation = soft_reflection_validation
        self.question_checks = [] if question_checks is None else question_checks
        self.case_insensitive = case_insensitive

        self.step = 0
        self.to_reflect = False
        self.finished = False
        self.answer = None
        self.soft_errors = []

    def openai_query(self, messages, tool_schemas, force_auto_tool_choice=False):
        args = {
            'model': self.model,
            'messages': messages
        }
        if len(tool_schemas) == 0:
            pass
        elif len(tool_schemas) == 1 and not force_auto_tool_choice:
            args['tool_choice'] = {'type': 'function', 'function': {'name': tool_schemas[0]['function']['name']}}
            args['tools'] = tool_schemas
        else:
            args['tool_choice'] = "auto"
            args['tools'] = tool_schemas

        completion = self.client.chat.completions.create( **args )
        return completion



    def query_and_process(self, tools=[], additional_info='', no_tool_calls_message=None):
        schemas = get_tool_defs(tools)
        response = self.openai_query(self.trace.to_messages(), schemas)
        message = response.choices[0].message
        self.trace.add_entry(message)
        results = process_response(response, tools)
        for result in results:
            if result.error is not None:
                raise self.LLMReactorError(result.error)
            reflection = self.clean_context_reflection(result)
            if len(additional_info) > 0:
                self.trace.add_entry({'role': 'system', 'content': additional_info})
        if len(schemas) > 0 and len(results) == 0:
            self.soft_errors.append("No function call")
            self.to_reflect = False
            if no_tool_calls_message is not None:
                message = { 'role': 'assistant', 'content': no_tool_calls_message }
                self.trace.add_entry(message)
        return results

    class ReflectionResult(BaseModel):
        relevant_quotes: List[str] = Field(..., description="A list of relevant quotes from the source that should be saved.")
        new_sources: List[str] = Field(..., description="A list of new links mentioned in the notes that should be checked later.")
        right_page: bool = Field(..., description="Are we on the right page?")
        comment: str = Field(..., description="A comment on the search results and next actions.")
        
        def refine_observation(self, observation: Observation):
            original_info_pieces = observation.info_pieces
            observation.clear_info_pieces()
            for quote in self.relevant_quotes:
                for info_piece in original_info_pieces:
                    if quote in info_piece.text:
                        info_piece.text = quote
                        observation.add_info_piece(info_piece)
            observation.interesting_links = self.new_sources
            observation.comment = self.comment
            observation.is_refined = True
            return observation


    def clean_context_reflection(self, result):
        if result.name == 'finish':
            result.tool = None
            self.trace.add_entry(result)
            return result
        elif result.name == "search":
            sysprompt = """
You are a researcher working on a user question. Previously you searched wikipedia and now
you need to evaluate the results. If you got any relevant information you need to extract
quotes quotes to be used as supporting evidence for answering the question.
You need to decide if you have enough information to answer the question and if you should look for more information on the same page or
or or if you need to follow a link or do another search.
"""
        elif result.name == "lookup" or result.name == "next_lookup":
            sysprompt = """
You are a researcher working on a user question. Previously you searched wikipedia and found a page.
You than did a keyword lookup on that page and now you need to evaluate the results.
If you got any relevant information you need to extract quotes quotes to be used as supporting evidence for answering the question.
You need to decide if you have enough information to answer the question and if you should look for more information on the same page or
or or if you need to follow a link or do another search.
"""
        elif result.name == "read_chunk":
            sysprompt = """
You are a researcher working on a user question. Previously you searched wikipedia and found a page.
You than retrieved another part of that page and now you need to evaluate the results.
If you got any relevant information you need to extract quotes quotes to be used as supporting evidence for answering the question.
You need to decide if you have enough information to answer the question and if you should look for more information on the same page or
or or if you need to follow a link or do another search.
"""
        elif result.name == "follow_link":
            sysprompt = """
You are a researcher working on a user question. Previously you followed a link to a page.
Now you need to evaluate the results of that page retrieval.
If you got any relevant information you need to extract quotes quotes to be used as supporting evidence for answering the question.
You need to decide if you have enough information to answer the question and if you should look for more information on the same page or
or or if you need to follow a link or do another search.
"""

        user_prompt = f"Here are the results for the search in Markdown format:\n\n{str(result.output)}"
        user_prompt += f"\n\nFrom these notes please extract quotes and note new sources relevant to answering the following question: {self.trace.user_question}"
        messages = [
            {'role': 'system', 'content': sysprompt},
            {'role': 'user', 'content': user_prompt},
        ]
        response = self.openai_query(messages, get_tool_defs([self.ReflectionResult]))
        message = response.choices[0].message
        #self.trace.add_entry(message)
        results = process_response(response, [self.ReflectionResult])
        if len(results) > 1:
            self.soft_errors.append(f"More than one reflection result")
        new_result = results[0]
        if new_result.error is not None:
            raise self.LLMReactorError(new_result.error)
        reflection = new_result.output
        result.output = reflection.refine_observation(result.output)
        result.tool = None
        self.trace.add_entry(result)
        return results
            
            
    def process(self):
        self.analyze_question()
        while self.answer is None:
            self.step += 1
            if self.step == self.max_llm_calls + 1:
                tools = [self.finish]
            else:
                tools = self.toolbox
            if self.step == self.max_llm_calls:
                step_info = "\n\nThis was the last data you can get - in the next step you need to formulate your answer"
            else:
                step_info = f"\n\nThis was {self.step} out of {self.max_llm_calls} calls for data."
            no_tool_calls_message = "You did not ask for any data this time - but it still counts."
            self.query_and_process(tools, additional_info=step_info, no_tool_calls_message=no_tool_calls_message)
            self.to_reflect = True

#            if 'gpt-4' in self.model:
#                time.sleep(20)


    def analyze_question(self):
        for query in self.question_checks:
            question_check = { 'role': 'user', 'content': query }
            logger.info(str(question_check))
            self.trace.add_entry(question_check)
            self.query_and_process()

    def finish(self,
               answer: Annotated[str, "The answer to the user's question"],
               answer_short: Annotated[str, "A short version of the answer"],
               reasoning: Annotated[str, "The reasoning behind the answer. Think step by step. Mention all assumptions you make."],
               ambiguity: Annotated[Optional[str], "Have you found anything in the retrieved information that makes the question ambiguous? For example a search for some name can show that there are many different entities with the same name."] = None
    ):
        """
        Finish the task and return the answer.
        """
        answer = Answer(answer, answer_short, reasoning, ambiguity)
        self.answer = answer
        return answer


def get_answer(question, config, client: object):
    print("\n\n<<< Question:", question)
    tool_class = config['tool']
    tool = tool_class(chunk_size=config['chunk_size'])
    toolbox = get_toolset_tools(tool)
    sys_prompt = PROMPTS[config['prompt_class']]
    initial_trace = Trace()
    initial_trace.add_system_message(sys_prompt(config['max_llm_calls'], ''))
    initial_trace.add_user_question(question)
    question_checks = QUESTION_CHECKS[config['question_check']] 
    reactor = LLMReactor(
        config['model'], toolbox, initial_trace, config['max_llm_calls'], client, None,
        question_checks=question_checks
    )
    reactor.process()
    return reactor

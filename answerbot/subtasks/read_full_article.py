import logging
from dataclasses import dataclass
from llm_easy_tools.processor import process_response
from llm_easy_tools.schema_generator import get_tool_defs
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import Field, BaseModel

READ_FULL_ARTICLE_SYSTEM_MESSAGE = '''
You are an assistant designed to help users extract key information and insights from documents. 
Your task is to read the provided document thoroughly, identify quotes that are relevant 
to answering the user's question, and summarize the relevant information. 
Ensure the quotes are directly related to the user's query and provide reasoning for their 
importance. Your summary should be concise and focused on addressing the question asked by the user.
'''

class Quote(BaseModel):
    quote: str = Field(description='The full text of the quote extracted from the document.')
    reasoning: str = Field(description='An explanation of why this quote is important and how it relates to the question.')

class ReadFullArticleReflection(BaseModel):
    relevant_quotes: list[Quote] = Field(description='A list of quotes from the document that are relevant to answering the question, along with explanations for their relevance.')
    summary: str = Field(description='A concise summary of the information from the document that is pertinent to answering the question.')



def read_full_article(
        client: OpenAI,
        model: str,
        question: str,
        document: str,
        document_title: str,
        context: str|None=None
        ):
    logging.getLogger(__name__).info(f'Reading Article: {document_title}')
    messages: list[ChatCompletionMessageParam] = [
            {'role': 'system', 'content': READ_FULL_ARTICLE_SYSTEM_MESSAGE},
            {'role': 'user', 'content': f'The document is titled as: {document_title}'},
            {'role': 'user', 'content': f'Question: {question}'},
            {'role': 'user', 'content': f'The text of the document:\n\n {document}'},
            ]

    if context:
        messages.append({'role': 'user', 'content': f'Additional context:\n {context}'})

    response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=get_tool_defs([ReadFullArticleReflection]),
            tool_choice={
                'type': 'function', 'function': {'name': 'ReadFullArticleReflection'}
                }
            )

    for call in process_response(response, [ReadFullArticleReflection]):
        if isinstance(call.output, ReadFullArticleReflection):
            return call.output

    raise RuntimeError('Expected a tool call')


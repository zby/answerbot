from dataclasses import dataclass
from typing import Annotated, Any, Callable, Collection
from llm_easy_tools.processor import process_response
from llm_easy_tools.schema_generator import get_function_schema, get_tool_defs
from openai import OpenAI
from openai.types.chat import (
        ChatCompletionMessageParam, 
        ChatCompletionNamedToolChoiceParam,
)
from pydantic import BaseModel, Field, annotated_handlers

from answerbot.subtasks.read_full_article import Quote, ReadFullArticleReflection, read_full_article
import logging


SHOW_TABLE_OF_CONTENTS_SYSTEM_MESSAGE = """
You are an LLM agent tasked with identifying relevant articles from a provided table of contents 
to help answer a given question. You will be given a table of contents with nested chapters and 
articles, a specific question, and some additional context. Your goal is to return a list of 
article IDs that are relevant to answering the question.

Instructions:

1. Analyze the provided table of contents, question, and any additional context.
2. Identify the articles that are most relevant to answering the question.
3. Limit the number of article IDs to the specified limit
4. For each article ID you choose, provide a reasoning explaining why that article is relevant 
to the question.

The table of contents is structured as follows:

Document Title
  Chapter Title
    [article id] Article Title
    [article id] Article Title

Example:

EU Artificial Intelligence Act
  Chapter I: General Provisions
    [kd18jpzr] Article 1: Subject Matter
    [mh0drpp0] Article 2: Scope

Please ensure your responses are concise, relevant, and well-justified based on the given context.
"""


@dataclass(frozen=True)
class DocumentSection:
    title: str
    children: list['DocumentSection|DocumentArticle']

    def find_article(self, id: str) -> 'DocumentArticle|None':
        for child in self.children:
            if isinstance(child, DocumentSection) and (found:=child.find_article(id)):
                return found
            if isinstance(child, DocumentArticle) and child.id == id:
                return child


    def to_string(self, depth: int=0) -> str:
        result = ('\t'*depth) + self.title
        if self.children:
            result += '\n' + '\n'.join(child.to_string(depth+1) for child in self.children)
        return result



@dataclass(frozen=True)
class DocumentArticle:
    title: str
    id: str
    retrieve: Callable[[], str]

    def to_string(self, depth: int=0) -> str:
        return ('\t'*depth) + f'[{self.id}] {self.title}'


class ArticleChoice(BaseModel):
    id: str = Field(description='The id of the page')
    reasoning: str = Field(description='Why is this page relevant to answering the question')


class ArticleChoices(BaseModel):
    choices: list[ArticleChoice] = Field(description='List of pages relevant to answering the questions')


@dataclass(frozen=True)
class ReadFromTableOfContentsArticleResult:
    article_choice: ArticleChoice
    article: DocumentArticle
    reflection: ReadFullArticleReflection


@dataclass(frozen=True)
class ReadFromTableOfContentsResult:
    articles: list[ReadFromTableOfContentsArticleResult]
    budget_spent: int

    def __str__(self) -> str:
        chunks = []

        for article in self.articles:
            chunks.append(
                    f'from "{article.article.title}": \n {article.reflection.summary}'
                    )
        return '\n\n'.join(chunks)


def show_table_of_contents(
        client: OpenAI,
        model: str,
        question: str,
        context: str,
        limit: int,
        document_toc: DocumentSection,
        ) -> list[ArticleChoice]:
    logging.getLogger(__name__).info(f'Opening table of contents for {document_toc.title}')

    messages: list[ChatCompletionMessageParam] = [
            {'role': 'system',  'content': SHOW_TABLE_OF_CONTENTS_SYSTEM_MESSAGE},
            {'role': 'user', 'content': f'Provide at most {limit} ids'},
            {'role': 'user', 'content': f'Table of contents:\n{document_toc.to_string()}'},
            {'role': 'user', 'content': f'Question: {question}'}
            ]
    if context:
        messages.append({
            'role': 'user', 'content': f'Additional context: {context}'
            })

    response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=get_tool_defs([ArticleChoices]),
            tool_choice=ChatCompletionNamedToolChoiceParam(
                type='function',
                function={'name': 'ArticleChoices'}
                )
            )

    tool_calls = process_response(response, [ArticleChoices])

    result = []

    for tool_call in tool_calls:
        if not isinstance(tool_call.output, ArticleChoices):
            continue
        result.extend(tool_call.output.choices)

    return result



class ReadFromTableOfContents:
    def __init__(
            self,
            client: OpenAI,
            model: str,
            question: str,
            document_toc: DocumentSection,
            read_article_cost=1,
            show_table_of_contents_cost=1,
            ) -> None:
        self._client = client
        self._model=model
        self._question = question
        self._document_toc = document_toc
        self._read_article_cost = read_article_cost
        self._show_table_of_contents_cost = show_table_of_contents_cost
        self._articles_read = set()

    def __call__(
            self,
            budget: Annotated[int, 'Budget you have left'],
            goal: Annotated[str, 'The information you expect to obtain'],
            context: Annotated[str|None, 'Any additional information gathered so far if any']):
        result = read_from_table_of_contents(
                client=self._client,
                model=self._model,
                question=self._question,
                read_article_cost=self._read_article_cost,
                show_table_of_contents_cost=self._show_table_of_contents_cost,
                budget=budget,
                context=context or '',
                articles_read=self._articles_read,
                document_toc=self._document_toc
                )
        self._articles_read.update({a.article_choice.id for a in result.articles})
        return result



def read_from_table_of_contents(
        client: OpenAI,
        model: str,
        question: str,
        context: str,
        budget: int,
        read_article_cost: int,
        show_table_of_contents_cost: int,
        document_toc: DocumentSection,
        articles_read: set[str],
        ) -> ReadFromTableOfContentsResult:
    budget_left = budget

    article_choices = show_table_of_contents(
            client,
            model,
            question=question,
            context=context,
            limit=budget // read_article_cost,
            document_toc=document_toc
            )

    article_choices = [
            choice for choice in article_choices
            if choice.id not in articles_read
            ]

    reflections = []

    budget_left -= show_table_of_contents_cost

    for choice in article_choices:
        if budget_left < read_article_cost:
            break
        article = document_toc.find_article(choice.id)
        if article is None:
            logging.getLogger(__name__).error('Could not find article with id=%s', choice.id)
            continue
        try:
            document = article.retrieve()
        except Exception as e:
            logging.getLogger(__name__).exception(e)
            continue


        reflection = read_full_article(
            client,
            model,
            question=question,
            document=document,
            document_title=f'{article.title} (from {document_toc.title})'                    
            )

        reflections.append(
                ReadFromTableOfContentsArticleResult(
                    article_choice=choice,
                    article=article,
                    reflection=reflection
                    )
                )

        budget_left -= read_article_cost


    return ReadFromTableOfContentsResult(
            articles=reflections,
            budget_spent=budget-budget_left,
            )


from dataclasses import dataclass
import re
from typing import Collection, Iterable, TypeAlias
from llm_easy_tools.processor import process_response
from llm_easy_tools.schema_generator import get_tool_defs
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field
import random
import string


NEWLINES_RE = re.compile(r'\n\n+')


@dataclass(frozen=True)
class CompressedParagraph:
    short_id: str
    full_text: str
    short_text: str


@dataclass(frozen=True)
class FullTextParagraph:
    short_id: str
    full_text: str


Paragraph: TypeAlias = CompressedParagraph|FullTextParagraph


SKIM_GET_RELEVANT_PARAGRAPHS_SYSTEM_PROMPT = '''
You are an intelligent assistant designed to extract and identify relevant paragraphs from a provided document based on a given question. Your task is to identify paragraphs that might be helpful in answering the question, considering the content and context of the document.

The document's paragraphs are provided in a specific format, where some paragraphs may be compressed. Compressed paragraphs contain ellipses ("...") indicating omitted parts, e.g., "mary had ... lamb ...". Fully expanded paragraphs do not contain any ellipses.

Please follow these steps:

1. Analyze the provided compressed text of the document, where each paragraph is represented by a short ID and may contain ellipses indicating compressed sections.
2. Identify the most relevant paragraphs that may help in answering the question.
3. For each relevant paragraph, provide its short ID and reasoning for its relevance.

Ensure that:
- Paragraphs without ellipses (i.e., fully expanded paragraphs) are included directly in the relevant quotes.
- Only paragraphs with ellipses (i.e., compressed paragraphs) are included in the list of paragraphs to expand.
- Expanding paragraphs is an expensive operation, so avoid requesting expansion if enough information can be obtained from the context and fully expanded paragraphs.

Examples:
- A compressed paragraph: "The quick brown fox ... lazy dog."
- A fully expanded paragraph: "The quick brown fox jumps over the lazy dog."

Format your response as a SkimReflections object containing:
- `paragraphs_to_expand`: List of compressed paragraphs that need to be expanded to answer the question.
- `relevant_expanded`: Relevant quotes from fully expanded paragraphs.
- `new_information_summary`: Short summary of the new information relevant to answering the question.
- `information_summary`: Summary of both the context and the new information relevant to answering the question.

Remember to:
- Understand and interpret the compressed format of paragraphs.
- Provide clear and concise reasoning.
- Select paragraphs that directly or indirectly relate to the question.
- Minimize the number of paragraphs to expand to those absolutely necessary.


Use SkimReflections tool to provide the result.
'''


class SkimReflection(BaseModel):
    paragraph_id: str = Field(description='The ID of the paragraph')
    reasoning: str = Field(description='Explanation of why this paragraph could be relevant to answering the question')

class CompressedReflectionQuote(BaseModel):
    quote: str = Field(description='The text of the relevant quote')
    reasoning: str = Field(description='Explanation of why this quote is relevant for answering the question')


class SkimReflections(BaseModel):
    paragraph_expansion_requests: list[SkimReflection] = Field(description='List of ellipsed paragraphs that need to be expanded to answer the question. Only include paragraphs marked as "compressed"')
    relevant_quotes: list[CompressedReflectionQuote] = Field(description='List of relevant quotes from already expanded paragraphs')
    new_information_summary: str = Field(description='Short summary of the new information relevant to answering the question')
    partial_answer: str = Field(
            description='From the provided context and the information gathered from the document, '
            "try to provide an answer to the user's question. If you can't give an answer just yet, "
            "provide all information that another agent could use to provide such an answer with "
            "more data"
            )



def skim_get_relevant_paragraphs(
        client: OpenAI,
        model: str,
        question: str,
        paragraphs: list[Paragraph],
        document_title: str,
        limit: int,
        context: str|None=None,
        ) -> SkimReflections|None:
    paragraphs_text = _paragraphs_text(paragraphs)
    expanded_paragraphs_ids = ', '.join(p.short_id for p in paragraphs if isinstance(p, FullTextParagraph))

    messages: list[ChatCompletionMessageParam]  = [
            {'role': 'system', 'content': SKIM_GET_RELEVANT_PARAGRAPHS_SYSTEM_PROMPT},
            {'role': 'user', 'content': f'provide at most {limit} ids'},
            {'role': 'user', 'content': f'The document is titled as: {document_title}'},
            {'role': 'user', 'content': f'The compressed text of the document:\n{paragraphs_text}'},
            {'role': 'user', 'content': f'Question: {question}'},
            ]
    if context:
        messages.append({'role': 'user', 'content': f'Additional context:\n{context}'})

    response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=get_tool_defs([SkimReflections]),
            tool_choice={'type': 'function', 'function': {'name': 'SkimReflections'}}
            )
    calls = process_response(response, [SkimReflections])

    for call in calls:
        if not isinstance(call.output, SkimReflections):
            continue
        return call.output

    return None


def split_into_paragraphs(markdown: str, max_words: int=24) -> list[Paragraph]:
    id_ = 0

    paragraphs = NEWLINES_RE.split(markdown)

    result = []

    for paragraph in paragraphs:
        if len(paragraph.split()) <= max_words:
            short_text = paragraph
        else:
            short_text = ' ... '.join(_compress_text(paragraph)) + ' ...'
        id_ += 1
        if short_text == paragraph:
            result.append(
                    FullTextParagraph(short_id=str(id_), full_text=paragraph)
                    )
        else:
            result.append(
                    CompressedParagraph(
                        short_id=str(id_),
                        full_text=paragraph,
                        short_text=short_text
                        )
                    )
    return result


def expand_paragraphs(paragraphs: Iterable[Paragraph], ids: Collection[str]) -> list[Paragraph]:
    result = []
    for paragraph in paragraphs:
        if paragraph.short_id in ids:
            result.append(
                    FullTextParagraph(
                        short_id=paragraph.short_id,
                        full_text=paragraph.full_text,
                        )
                    )
        else:
            result.append(paragraph)
    return result


def _paragraphs_text(paragraphs: list[Paragraph]) -> str:
    chunks = []

    for paragraph in paragraphs:
        if isinstance(paragraph, FullTextParagraph):
            chunks.append(f'{paragraph.full_text}')
        else:
            chunks.append(f'[{paragraph.short_id} (compressed paragraph)] {paragraph.short_text}')

    return '\n\n'.join(chunks)


def _compress_text(text: str, keep_min: int=2, keep_max: int=8, skip_min: int=2, skip_max: int=8, keep_ratio: float=0.05, skip_ratio: float=0.2) -> list[str]:
    words = text.split()
    nwords = len(words)

    keep_count = min(keep_max, max(keep_min, int(keep_ratio * nwords)))
    skip_count = min(skip_max, max(skip_min, int(skip_ratio*nwords)))

    result = []

    i = 0

    while i < nwords:
        result.append(' '.join(words[i:i+keep_count]))
        i += keep_count + skip_count

    return result



def _short_id():
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))

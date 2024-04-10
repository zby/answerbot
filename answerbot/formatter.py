from typing import Any, Callable
from pprint import pformat
import textwrap

from answerbot.prompt_builder import FunctionResult, FunctionCall

from answerbot.prompt_templates import Question


def format_markdown(prompts, width: int=70) -> str:

    question = None
    answer = None
    reasoning = None
    next_url = None
    content_pieces = []

    for prompt in prompts.parts:
        if type(prompt) is FunctionCall and prompt.name=='simplereflection_and_goto_url':
            next_url = prompt.args['url']
        if type(prompt) is Question:
            question = prompt.content
        if hasattr(prompt, 'args') and isinstance(prompt.args, dict) and 'answer' in prompt.args: 
            answer = prompt.args['answer']
            reasoning = prompt.args['reasoning']
        if prompt.content:
            content_pieces.append([prompt.content, False, prompt, next_url])
        if hasattr(prompt, 'args') and isinstance(prompt.args, dict) and 'is_relevant' in prompt.args:
            content_pieces[-1][1] = prompt.args['is_relevant']

    result = ''

    if question:
        result += '# Question\n'
        result += '\n'.join(textwrap.wrap(question, width=width))
        result += '\n\n'
    if answer is not None:
        result += '# Answer\n'
        result += '\n'.join(textwrap.wrap(answer, width=width))
        result += '\n\n'

    if reasoning:
        result += '# Reasoning\n'
        result += '\n'.join(textwrap.wrap(reasoning, width=70))
        result += '\n\n'

    if content_pieces:
        result += '# Retreived Information\n'
        for piece, is_relevant, prompt, url in content_pieces:
            piece = piece.rsplit('\n\nThis was')[0]
            if type(prompt) is not FunctionResult:
                continue
            if prompt.name != 'simplereflection_and_read_chunk':
                continue
            if not is_relevant:
                continue
            if url:
                result += f'\n**from**: <{url}>'
            result += f'\n> ' + '\n> '.join(textwrap.wrap(piece, width=width)) + '\n'
    return result


def _get_item(dct: dict[str, Any], *path: str) -> Any:
    for item in path:
        if not isinstance(dct, dict):
            return None
        if not isinstance(item, str):
            return None
        if item not in dct:
            return None
        dct = dct[item]
    return dct


def _predicate(dct: dict[str, Any], predicate: Callable[[Any], bool], *path):
    item = _get_item(dct, *path)
    try:
        return predicate(item)
    except:
        return False


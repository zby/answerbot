from dataclasses import dataclass
from functools import wraps
import inspect
import json
from time import time
import openai
import httpx
from typing import Annotated, Any, Callable, DefaultDict, Iterator, Self, TypeAlias, get_origin
import logging

from openai.resources.beta.threads import messages
from .schema import parameters_basemodel_from_function, get_function_openai_schema


from openai.types.chat import (
        ChatCompletionMessage,
        ChatCompletionMessageParam,
        ChatCompletionMessageToolCall, 
        ChatCompletionToolChoiceOptionParam, 
        ChatCompletionToolParam,
        )
from openai._types import NOT_GIVEN
import logging


class FunctionNotFoundError(Exception):
    pass


class FunctionArgumentsMalformedError(Exception):
    pass


@dataclass(frozen=True)
class ToolCallResult:
    tool_call: ChatCompletionMessageToolCall
    message: ChatCompletionMessage
    function: Callable|None = None
    arguments: dict[str, Any]|None = None
    result: Any = None
    exception: Exception|None = None

    def get_result(self):
        if self.function is None:
            raise FunctionNotFoundError(self.tool_call.function.name)
        if self.arguments is None:
            raise FunctionArgumentsMalformedError()
        if self.exception:
            raise self.exception
        return self.result


@dataclass(frozen=True)
class SucceededToolCallResult:
    tool_call: ChatCompletionMessageToolCall
    message: ChatCompletionMessage
    function: Callable
    arguments: dict[str, Any]
    result: Any



Message: TypeAlias = dict[str, Any]
Messages: TypeAlias = list[ChatCompletionMessage]



@dataclass(frozen=True)
class QueryResult:
    message: ChatCompletionMessage
    tool_calls: list[ToolCallResult]
    succeeded_calls: dict[Callable, list[SucceededToolCallResult]]

    def first_succeeded(self, function: Callable) -> SucceededToolCallResult|None:
        if function not in self.succeeded_calls:
            return None
        return self.succeeded_calls[function][0]
    


class LLM:
    def __init__(
            self, model: str, 
            client: openai.OpenAI|None=None,
            messages: Messages|None = None,
            throttle=10,
            ) -> None:
        self._client = client or openai.OpenAI(
                timeout=httpx.Timeout(70, read=60.0, write=20.0, connect=6.0)
                )
        self._model = model
        self.messages = [] if messages is None else messages
        self._logger = logging.getLogger(__name__)
        self._throttle=throttle
        self.queries_made = 0
        self._last_query = 0

    def copy(
            self, 
            model: str|None=None, 
            client: openai.OpenAI|None=None, 
            messages: Messages|None=None) -> Self:
        return LLM(
                model=self._model if model is None else model,
                client=self._client if client is None else client,
                messages=self.messages[:] if messages is None else messages
                )

    def query(
            self, 
            tools: list[Callable]|Callable=[], 
            ) -> QueryResult:
        while self._last_query+self._throttle > time():
            continue
        self._last_query = time()
        if callable(tools):
            assert hasattr(tools, '__name__')
            tool_choice = {
                    'type': 'function',
                    'function': {'name': tools.__name__}  # type: ignore
                    }
        else:
            tool_choice = 'auto'

        if callable(tools):
            tools_ = [get_function_openai_schema(tools)]
        elif tools:
            tools_ = list(map(get_function_openai_schema, tools))
        else:
            tools_ = NOT_GIVEN

        response = self._client.chat.completions.create(
                model=self._model,
                messages=self.messages,  # type: ignore
                tools=tools_,  # type: ignore
                tool_choice=tool_choice if tools else NOT_GIVEN,  # type: ignore
                )
        self.queries_made += 1
        message = response.choices[0].message
        self.messages.append(message)

        succeeded_calls = DefaultDict(list)
        tool_calls = []
        if callable(tools):
            functions = {tools.__name__: tools}
        else:
            functions = {f.__name__: f for f in tools} 

        for ct in (call_tool(tc, functions, self._logger) for tc in message.tool_calls or ()):
            self.messages.append(ct.message)
            tool_calls.append(ct)
            if ct.result is not None:
                succeeded_calls[ct.function].append(SucceededToolCallResult(
                        tool_call=ct.tool_call,
                        message=ct.message,
                        function=ct.function,  # type: ignore
                        arguments=ct.arguments,  # type: ignore
                        result=ct.result,
                        )
                        )

        return QueryResult(
                message=message,
                tool_calls=tool_calls,
                succeeded_calls=succeeded_calls,
                )

    def call_tools(
            self,
            message: ChatCompletionMessage,
            tools: list[Callable],
            ) -> Iterator[ToolCallResult]:
        functions = {f.__name__: f for f in tools}
        r = []
        for ct in (call_tool(tc, functions, self._logger) for tc in message.tool_calls or ()):
            self.messages.append(ct.message)
            r.append(ct)
        yield from r

    def add_message(self, message: dict[str, Any]):
        self.messages.append(message)  # type: ignore
        self._logger.debug(str(message))



def call_tool(tool_call: ChatCompletionMessageToolCall, functions: dict[str, Callable], logger: logging.Logger) -> ToolCallResult:
    #functions = {f.__name__: f for f in tools}

    try:
        args = json.loads(tool_call.function.arguments)
    except ValueError:
        args = None

    function_name = tool_call.function.name
    if function_name not in functions:
        logger.error(f'Funtion {function_name} not found')
        return ToolCallResult(
                tool_call=tool_call,
                message=function_result_to_message(tool_call, 'error: function does not exist'),
                arguments=args,
                )

    function = functions[function_name]

    if args is None:
        logger.error(f'Tried to call function {function_name} with malformed arguments')
        return ToolCallResult(
                tool_call=tool_call,
                message=function_result_to_message(tool_call, 'error: malformed arguments'),
                function=function
                )

    try:
        args_model = parameters_basemodel_from_function(function)
        args_model_instance = args_model.model_validate(args)
        result = function(**args_model_instance.model_dump())
    except Exception as e:
        logger.exception(e)
        return ToolCallResult(
                tool_call=tool_call,
                message=function_result_to_message(
                    tool_call,
                    {'error': {'type': str(type(e)), 'message': str(e)}},
                    ),
                function=function,
                arguments=args,
                exception=e,
                )
    logger.info(f'Called function {function_name} with arguments {args}')
    logger.info(f'result: {result}')
    return ToolCallResult(
            tool_call=tool_call,
            message=function_result_to_message(tool_call, result),
            function=function,
            arguments=args,
            result=result
            )


def function_result_to_message(tool_call, result) -> ChatCompletionMessage:
    return {
            'tool_call_id': tool_call.id,
            'role': 'tool',
            'name': tool_call.function.name,
            'content': json.dumps(result)
            }   # type: ignore



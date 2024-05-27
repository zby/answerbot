from typing import Any, Callable
from llm_easy_tools.processor import BaseModel, process_response
from llm_easy_tools.schema_generator import get_tool_defs
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

REACTOR_SYSTEM_MESSAGE = """
You are an AI model tasked with answering questions within a specified domain using a set of tools. 
Each tool has a specific cost mentioned in its docstring as 'COST: x'. You will have a limited budget
to answer the question, so choose your tools wisely.
Some tools may also have usage limits.

Your goal is to provide a complete answer to the following question:

Domain: {domain}

Question: {question}

Your budget for this task is: {energy}

Manage your energy carefully by selecting appropriate tools based on their energy costs. 
If you reach an acceptable answer or run out of energy, use the 'Finish' tool to conclude.

Information gathered so far: {context}

Please proceed by selecting the most appropriate tools to answer the question effectively 
while managing your energy.
"""



class Finish(BaseModel):
    answer: str


def use_limit(limit: int):
    def _decorator(f):
        f.__reactor_use_limit__ = limit
        return f
    return _decorator


def cost(cost: int):
    def _decorator(f):
        f.__reactor_cost__ = cost
        f.__doc__ = (getattr(f, '__doc__', '') or '') + f'\n\nCOST: {cost}'
        return f
    return _decorator


def llm_react(
        client: OpenAI,
        model: str,
        budget: int,
        domain: str,
        question: str,
        toolset: list[Callable],
        ) -> list[Any]:
    budget_left = budget
    result = []
    tools_used = {}
    context = ''
    done = False

    while budget_left > 0 and not done:
        toolset_ = toolset + [Finish]
        toolset_ = [
                tool
                for tool in toolset_
                if getattr(tool, '__reactor_cost__', 0) <= max(0, budget_left)
                and getattr(tool, '__reactor_use_limit__', float('inf')) > tools_used.get(tool, 0)
                ]

        message: ChatCompletionMessageParam = {
                'role': 'system',
                'content': REACTOR_SYSTEM_MESSAGE.format(
                    domain=domain,
                    energy=budget_left,
                    context=context,
                    question=question,
                    )
                }

        response = client.chat.completions.create(
                model=model,
                messages=[message],
                tools=get_tool_defs(toolset_),
                tool_choice=(
                    'auto' if len(toolset_) > 1
                    else get_tool_defs(toolset_)[0]
                    )
                )

        for call in process_response(response, toolset_):
            if isinstance(call.output, Finish):
                done = True
            else:
                context += str(call.output) + '\n\n'
            result.append(call.output)
            budget_left -= (
                    getattr(call.output, 'energy_spent', 0) 
                    + getattr(call.tool, '__reactor_cost__', 0)
                    )
            tools_used[call.tool] = tools_used.get(call.tool, 0) + 1

    return result
   

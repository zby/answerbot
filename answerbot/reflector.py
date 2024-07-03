from dataclasses import dataclass
from answerbot.trace import Trace, KnowledgeBase
from answerbot.tools.observation import Observation
from answerbot.clean_reflection import ReflectionResult

from llm_easy_tools import process_response, get_tool_defs

@dataclass(frozen=True)
class Reflector:
    system_prompt: str = "You are a researcher working on a user question in a team with other researchers. You need to check the assumptions that the other researchers made."
    user_prompt_template: str = """# Question

The user's question is: {question}{learned_stuff}

# Retrieval

We have performed information retrieval with the following results:
{observation}

# Current task

You need to review the information retrieval recorded above and reflect on it.
You need to note all information that can help in answering the user question that can be learned from the retrieved fragment,
together with the quotes that support them.
Please remember that the retrieved content is only a fragment of the whole page.
"""

    def reflect(self, model: str, observation: Observation, question: str, knowledge_base: KnowledgeBase) -> None:
        trace = Trace()
        learned_stuff = f"\n\nSo far we have some notes on the following urls:{knowledge_base.learned()}" if not knowledge_base.is_empty() else ""
        user_prompt = self.user_prompt_template.format(question=question, learned_stuff=learned_stuff, observation=str(observation))
        trace.append({'role': 'system', 'content': self.system_prompt})
        trace.append({'role': 'user', 'content': user_prompt})
        schemas = get_tool_defs([ReflectionResult])
        response = trace.openai_query(model, schemas)
        results = process_response(response, [ReflectionResult])
        new_result = results[0]
        trace.append(new_result)
        if new_result.error is not None:
            raise new_result.error
        reflection = new_result.output
        reflection_string = self.update_knowledge_base(knowledge_base, reflection, observation)
        trace.hidden_result = reflection_string
        print(reflection_string)
        return trace

    def update_knowledge_base(self, knowledge_base: KnowledgeBase, reflection: ReflectionResult, observation: Observation) -> str:
        knowledge_piece = reflection.extract_knowledge(observation)
        knowledge_base.add_knowledge_piece(knowledge_piece)
        reflection.remove_checked_urls(knowledge_base.urls())
        reflection_string = f"current url: {knowledge_piece.url}\n"
        if len(reflection.new_sources) > 0 or not knowledge_piece.is_empty():
            reflection_string += f"{str(knowledge_piece)}\n"
            if len(reflection.new_sources) > 0:
                reflection_string += f"Discovered new sources: {reflection.new_sources}"
        return reflection_string


@dataclass(frozen=True)
class Planner:
    system_prompt: str = "You are a researcher working on a user question in a team with other researchers. You need to check the assumptions that the other researchers made."
    planning_prompt_template: str = """# Question

The user's question is: {question}

# Available tools

{available_tools}

# Retrieval

We have performed information retrieval with the following results:

{observation}

# Reflection

{reflection}

# Next step

What would you do next?
Please analyze the retrieved data and check if you have enough information to answer the user question.

If you still need more information, consider the available tools.

You need to decide if the current page is relevant to answer the user question.
If it is, then you should recommed exploring it further with the `lookup` or `read_more` tools.

When using `search` please use simple queries. When trying to learn about a property of an object or a person,
first search for that object then you can browse the page to learn about its properties.
For example to learn about the nationality of a person, first search for that person.
If the persons page is retrieved but the information about nationality is not at the top of the page
you can use `read_more` to continue reading or call `lookup('nationality')` or `lookup('born')` to get more information.

Please specify both the tool and the parameters you need to use if applicable.
Explain your reasoning."""


    def plan_next_action(self, model, reflection_string: str, observation: Observation, question: str) -> Trace:
        trace = Trace()
        sysprompt = self.system_prompt
        user_prompt = self.planning_prompt_template.format(
            question=question,
            available_tools=observation.available_tools,
            observation=str(observation),
            reflection=reflection_string
        )
        trace.append({'role': 'system', 'content': sysprompt})
        trace.append({'role': 'user', 'content': user_prompt})

        response = trace.openai_query(model)
        planning_result = response.choices[0].message.content

        planning_string = f"**My Notes**\n{reflection_string}\n\nHmm what I could do next?\n\n{planning_result}"
        trace.result = {'role': 'user', 'content': planning_string}

        return trace


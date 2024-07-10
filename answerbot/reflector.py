from dataclasses import dataclass
from answerbot.clean_reflection import KnowledgeBase, KnowledgePiece
from answerbot.tools.observation import Observation, InfoPiece
from answerbot.clean_reflection import ReflectionResult

from answerbot.chat import Chat, Prompt
from answerbot.qa_prompts import SystemPrompt, prompt_templates


@dataclass(frozen=True)
class LearningPrompt(Prompt):
    memory: KnowledgeBase
    question: str
    observation: Observation


learning_templates: dict[str, str] = {
    SystemPrompt: "You are a researcher working on a user question in a team with other researchers. You need to check the assumptions that the other researchers made.",
    LearningPrompt: """# Question

The user's question is: {{question}}
{% if not memory.is_empty() %}

# Notes from previous work

We have some notes on the following urls:
{{ memory.learned() }}
{% endif %}

# Retrieval

We have performed information retrieval with the following results:
{{observation}}

# Current task

You need to review the information retrieval recorded above and reflect on it.
You need to note all information that can help in answering the user question that can be learned from the retrieved fragment,
together with the quotes that support them.
Please remember that the retrieved content is only a fragment of the whole page."""
}

learning_templates = prompt_templates | learning_templates


def reflect(model: str, observation: Observation, question: str, knowledge_base: KnowledgeBase) -> str:
    chat = Chat(
        model=model,
        one_tool_per_step=True,
        system_prompt=SystemPrompt(),
        templates=learning_templates
    )
    chat.append(LearningPrompt(memory=knowledge_base, question=question, observation=observation))

    reflections = []
    for reflection in chat.process([ReflectionResult]):
        if observation.current_url:
            reflections.append(knowledge_base.update_knowledge_base(reflection, observation))
    reflection_string = '\n'.join(reflections)

    return reflection_string


@dataclass(frozen=True)
class PlanningPrompt(Prompt):
    question: str
    available_tools: str
    observation: Observation
    reflection: str

planning_templates = {
    SystemPrompt: "You are a researcher working on a user question in a team with other researchers. You need to check the assumptions that the other researchers made.",
    PlanningPrompt: """# Question

The user's question is: {{question}}

# Available tools

{{available_tools}}

# Retrieval

We have performed information retrieval with the following results:

{{observation}}

# Reflection

{{reflection}}

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
}

def plan_next_action(model: str, observation: Observation, question: str, reflection_string: str) -> str:
    chat = Chat(
        model=model,
        system_prompt=SystemPrompt(),
        templates=planning_templates
    )

    planning_prompt = PlanningPrompt(
        question=question,
        available_tools=observation.available_tools,
        observation=observation,
        reflection=reflection_string
    )

    chat.append(planning_prompt)

    response = chat.llm_reply()
    planning_result = response.choices[0].message.content

    planning_string = f"**My Notes**\n{reflection_string}\n\nHmm what I could do next?\n\n{planning_result}"

    return planning_string



if __name__ == "__main__":
    # Example usage of reflect and plan_next_action functions
    from dotenv import load_dotenv
    import litellm

    load_dotenv()
    litellm.success_callback = ["langfuse"]
    litellm.failure_callback = ["langfuse"]

    # Set up example data
    model = "gpt-3.5-turbo"  # or any other model you're using
    question = "What was the capital of the East Roman Empire?"

    # Create an example Observation
    observation = Observation(
        current_url="https://en.wikipedia.org/wiki/Byzantine_Empire",
        info_pieces=[
            InfoPiece("Constantinople was the capital of the Byzantine Empire.", source="Wikipedia", quotable=True),
            InfoPiece("The Byzantine Empire lasted from 330 AD to 1453 AD.", source="History.com", quotable=True)
        ],
        available_tools="search, lookup, read_more",
        operation="Initial search"
    )

    # Example knowledge base
    what_have_we_learned = KnowledgeBase()
    new_knowledge_piece = KnowledgePiece(
        url="https://www.britannica.com/place/Byzantine-Empire",
        quotes=["The Byzantine Empire was also known as the Eastern Roman Empire and was a continuation of the Roman Empire in its eastern provinces."],
        learned="The Byzantine Empire was a continuation of the Roman Empire in its eastern provinces and was also known as the Eastern Roman Empire."
    )

    # Add the new KnowledgePiece to the knowledge base
    what_have_we_learned.add_knowledge_piece(new_knowledge_piece)

    # Call reflect function
    reflection_string = reflect(model, observation, question, what_have_we_learned)
    print("Reflection:")
    print(reflection_string)
    print("\n" + "="*50 + "\n")

    # Call plan_next_action function
    planning_string = plan_next_action(model, observation, question, reflection_string)
    print("Planning:")
    print(planning_string)

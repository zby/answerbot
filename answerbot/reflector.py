from dataclasses import dataclass
from typing import Optional
from pydantic import BaseModel, Field, field_validator
from fuzzywuzzy import fuzz

from answerbot.knowledgebase import KnowledgeBase, KnowledgePiece
from answerbot.tools.observation import Observation, InfoPiece

from answerbot.chat import Chat, Prompt, SystemPrompt


class ReflectionResult(BaseModel):
    what_have_we_learned: Optional[str] = Field(..., description="Have we learned anything that would help us answer the user question from the retrieved information and why?")
    comment: str = Field(..., description="A comment on the retrieved information.")
    relevant_quotes: list[str] = Field(..., description="A list of relevant literal quotes from the source that should be saved.")
    new_sources: list[str] = Field(..., description="A list of new urls mentioned in the notes that should be checked later.")

    @field_validator('new_sources', mode='before')
    def unique_new_sources(cls, v):
        if v is None:
            return []
        return list(dict.fromkeys(v))

    def refine_observation(self, observation: Observation):
        original_info_pieces = observation.info_pieces
        observation.clear_info_pieces()
        for info_piece in original_info_pieces:
            if info_piece.quotable:
                for quote in self.relevant_quotes:
                    if quote in info_piece.text:
                        info_piece.text = quote
                        observation.add_info_piece(info_piece)
            else:
                observation.add_info_piece(info_piece)
        observation.interesting_links = self.new_sources
        observation.comment = self.comment
        observation.is_refined = True
        return observation
    
    def __str__(self):
        content = ''
        if self.relevant_quotes:
            quotes_string = "".join("\n > " + quote for quote in self.relevant_quotes)
            content += f"Here are quotes that look relevant:{quotes_string}\n\n"
        if self.new_sources:
            new_sources_string = "".join("\n - " + link for link in self.new_sources)
            content += f"Some links from the notes that might contain relevant information that we should check later:\n{new_sources_string}\n"
        if len(self.comment) > 0:
            content += f"{self.comment}"
        return content

    def remove_checked_urls(self, urls: list[str]):
        for url in urls:
            if url in self.new_sources:
                self.new_sources.remove(url)

    def extract_knowledge(self, observation:Observation):
        checked_quotes = []
        for quote in self.relevant_quotes:
            for info_piece in observation.info_pieces:
                if not info_piece.quotable:
                    continue
                # Using partial ratio for approximate substring matching
                if fuzz.partial_ratio(quote, info_piece.text) > 80:  # adjust the threshold as needed
                    checked_quotes.append(quote)
        return KnowledgePiece(url=observation.current_url, quotes=checked_quotes, learned=self.what_have_we_learned)

    def update_knowledge_base(self, knowledge_base: KnowledgeBase, observation: Observation) -> str:
        knowledge_piece = self.extract_knowledge(observation)
        knowledge_base.add_knowledge_piece(knowledge_piece)
        self.remove_checked_urls(knowledge_base.urls())
        reflection_string = f"current url: {knowledge_piece.url}\n"
        if len(self.new_sources) > 0 or not knowledge_piece.is_empty():
            reflection_string += f"{str(knowledge_piece)}\n"
            if len(self.new_sources) > 0:
                reflection_string += f"Discovered new sources: {self.new_sources}"
        return reflection_string



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
            reflections.append(reflection.update_knowledge_base(knowledge_base, observation))
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

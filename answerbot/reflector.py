from dataclasses import dataclass
from typing import Optional, Callable
from pydantic import BaseModel, Field, field_validator
from fuzzywuzzy import fuzz

from llm_easy_tools import LLMFunction, get_tool_defs 

from answerbot.knowledgebase import KnowledgeBase, KnowledgePiece
from answerbot.tools.observation import Observation, InfoPiece

from answerbot.chat import Chat, HasLLMTools, expand_toolbox
from answerbot.qa_prompts import PlanningPrompt, PlanningSystemPrompt, ReflectionPrompt, ReflectionSystemPrompt

import logging


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



def reflect(model: str, prompt_templates: dict, question: str, observation: Observation, knowledge_base: KnowledgeBase) -> str:
    chat = Chat(
        model=model,
        one_tool_per_step=True,
        system_prompt=ReflectionSystemPrompt(),
        templates=prompt_templates
    )
    chat.append(ReflectionPrompt(memory=knowledge_base, question=question, observation=observation))

    reflections = []
    for reflection in chat.process([ReflectionResult]):
        if observation.current_url:
            reflections.append(reflection.update_knowledge_base(knowledge_base, observation))
    reflection_string = '\n'.join(reflections)

    return reflection_string


def format_tool_docstrings(schemas: list[dict]) -> str:
    formatted_list = []
    for schema in schemas:
        func_name = schema['function']['name']
        description = schema['function'].get('description', '')

        # Start with function name and description
        doc = f"- **{func_name}**\n\n"
        doc += "\n".join(f"  {line}" for line in description.split('\n')) + "\n\n"

        # Add parameters section if present
        if 'parameters' in schema['function']:
            doc += "  Parameters\n  ----------\n"
            properties = schema['function']['parameters'].get('properties', {})
            for param, details in properties.items():
                param_type = details.get('type', 'Any')
                param_desc = details.get('description', '')
                doc += f"  {param} : {param_type}\n"
                doc += "\n".join(f"      {line}" for line in param_desc.split('\n')) + "\n"

        formatted_list.append(doc)

    return "\n".join(formatted_list)


def plan_next_action(
    model: str,
    prompt_templates: dict,
    question: str,
    available_tools: list[LLMFunction, Callable, HasLLMTools],
    observation: Optional[Observation] = None,
    reflection_string: Optional[str] = None
) -> str:
    chat = Chat(
        model=model,
        system_prompt=PlanningSystemPrompt(),
        templates=prompt_templates
    )

    schemas = get_tool_defs(expand_toolbox(available_tools))
    available_tools_str = format_tool_docstrings(schemas)

    planning_prompt = PlanningPrompt(
        question=question,
        available_tools=available_tools_str,
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
    
    from answerbot.qa_prompts import wiki_researcher_prompts

    from dotenv import load_dotenv
    import litellm
    import sys

    load_dotenv()
    litellm.success_callback = ["langfuse"]
    litellm.failure_callback = ["langfuse"]

    # Configure logging for the chat module
    #chat_logger = logging.getLogger('answerbot.chat')
    #chat_logger.setLevel(logging.DEBUG)
    #chat_logger.addHandler(logging.StreamHandler(sys.stdout))


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
    reflection_string = reflect(model, wiki_researcher_prompts, question, observation, what_have_we_learned)
    print("Reflection:")
    print(reflection_string)
    print("\n" + "="*50 + "\n")

    # Define available tools
    def search(query: str) -> str:
        """
        Search for information on a given query.
        """
        pass

    def lookup(keyword: str) -> str:
        """
        Look up a specific keyword in the current document.
        """
        pass

    def read_more() -> str:
        """
        Continue reading the current document from where it was left off.
        """
        pass

    available_tools = [search, lookup, read_more]


    # Call plan_next_action function
    planning_string = plan_next_action(model, wiki_researcher_prompts, question, available_tools, observation, reflection_string)
    print("Planning:")
    print(planning_string)

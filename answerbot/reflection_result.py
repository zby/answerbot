from dataclasses import dataclass
from typing import Optional, Callable
from pydantic import BaseModel, Field, field_validator

from answerbot.knowledgebase import KnowledgeBase, KnowledgePiece
from answerbot.tools.observation import Observation

import logging

import re
from difflib import SequenceMatcher

def find_similar_fragments(text, quote):
    # Convert quote to lowercase for case-insensitive matching
    quote_lower = quote.lower()

    # Create a pattern that allows for up to 5 non-word, non-whitespace characters between words
    pattern = r'[^\w\s]{0,5}\b' + r'\W{0,5}'.join(re.escape(word) for word in quote_lower.split()) + r'\b[^\w\s]{0,5}'

    # Find all matches, case-insensitive
    matches = re.finditer(pattern, text, re.IGNORECASE)

    # Extract the exact matches from the original text
    similar_fragments = [text[m.start():m.end()] for m in matches]

    return similar_fragments

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
        refined_content = ""
        for quote in self.relevant_quotes:
            if quote in observation.content:
                refined_content += quote + "\n"

        return Observation(
            content=refined_content.strip(),
            source=observation.source,
            operation=observation.operation,
            quotable=True
        )

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

    def extract_knowledge(self, observation: Observation):
        checked_quotes = []
        for quote in self.relevant_quotes:
            similar_fragments = find_similar_fragments(observation.content, quote)
            checked_quotes.extend(similar_fragments)
        return KnowledgePiece(url=observation.source, quotes=checked_quotes, learned=self.what_have_we_learned)

    def update_knowledge_base(self, knowledge_base: KnowledgeBase, observation: Observation) -> str:
        knowledge_piece = self.extract_knowledge(observation)
        knowledge_base.add_knowledge_piece(knowledge_piece)
        self.remove_checked_urls(knowledge_base.urls())
        reflection_string = f"current url: {knowledge_piece.url}\n"
        if len(self.new_sources) > 0 or not knowledge_piece.is_empty():
            reflection_string += f"{str(knowledge_piece)}\n"
            if len(self.new_sources) > 0:
                reflection_string += f"Discovered new sources: {self.new_sources}"
        reflection_string += f"\n\n{self.comment}"
        return reflection_string

if __name__ == "__main__":
    text = """Extended discussion on artificial *intelligence* and its impacts.
Brief mention of machine learning within broader tech trends.
Unrelated content about economics.
Unquotable.
Artificial intelligence could revolutionize many sectors."""
    print(find_similar_fragments(text, "Artificial intelligence"))
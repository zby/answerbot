from dataclasses import dataclass
from typing import Optional, Callable
from pydantic import BaseModel, Field, field_validator

from answerbot.tools.observation import Observation, KnowledgePiece, History

import logging

import re
from difflib import SequenceMatcher
MARKDOWN_LINK_PATTERN = r'\[([^\]]+)\]\(([^\)]+)\)'

def find_similar_fragments(text, quote):
    # Remove markdown links
    text_without_links = re.sub(MARKDOWN_LINK_PATTERN, r'\1 ', text)

    similar_fragments = []

    # Convert quote to lowercase for case-insensitive matching
    quote_lower = quote.lower()

    # Split the quote into sequences of word characters
    words = re.findall(r'\w+', quote_lower)
    escaped_words = [re.escape(word) for word in words]

    # Create a pattern that allows for up to 5 non-word, non-whitespace characters between words
    pattern = r'[^\w\s]{0,5}' + r'.{0,5}'.join(escaped_words) + r'[^\w\s]{0,5}'

    # Find all matches, case-insensitive
    for match in re.finditer(pattern, text_without_links, re.IGNORECASE):
        fragment = text_without_links[match.start():match.end()]
        similar_fragments.append(fragment)

    return similar_fragments

class Note(BaseModel):
    content: str = Field(..., description="What we learned from the Current Observation.")
    source: str = Field(..., description="A literal quote from the Current Observation that supports the content of the note.")

class ReflectionResult(BaseModel):
    what_have_we_learned: str = Field(..., description="Have we learned anything new in the Current Observation that would help us answer the user question and why?")
    comment: str = Field(..., description="A comment on the retrieved information.")
    relevant_quotes: list[str] = Field(..., description="A list of relevant literal quotes from the Current Observation that suppor the should be saved.")
    new_sources: list[str] = Field(..., description="A list of new urls mentioned in the Current Observation that should be checked later.")

    @field_validator('new_sources', mode='before')
    def unique_new_sources(cls, v):
        if v is None:
            return []
        return list(dict.fromkeys(v))

    def remove_checked_sources(self, sources: list[str]):
        for source in sources:
            if source in self.new_sources:
                self.new_sources.remove(source)

    def extract_knowledge(self, observation: Observation):
        checked_quotes = []
        if self.relevant_quotes is None:
            logging.warning("No relevant quotes found, creating empty list!!!")
            self.relevant_quotes = []
        for quote in self.relevant_quotes:
            similar_fragments = find_similar_fragments(observation.content, quote)
            checked_quotes.extend(similar_fragments)
        return KnowledgePiece(source=observation, quotes=checked_quotes, content=self.what_have_we_learned)

    def update_history(self, history: History) -> str:
        observation = history.observations[-1]
        knowledge_piece = self.extract_knowledge(observation)
        history.add_knowledge_piece(knowledge_piece)
        self.remove_checked_sources(history.sources())

if __name__ == "__main__":
    text = """Extended discussion on artificial *intelligence* and its impacts.
Brief mention of machine learning within broader tech trends.
Unrelated content about economics.
Unquotable.
Artificial intelligence could revolutionize many sectors."""
    print(find_similar_fragments(text, "Artificial intelligence"))
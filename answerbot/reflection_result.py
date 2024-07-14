from dataclasses import dataclass
from typing import Optional, Callable
from pydantic import BaseModel, Field, field_validator
from fuzzywuzzy import fuzz


from answerbot.knowledgebase import KnowledgeBase, KnowledgePiece
from answerbot.tools.observation import Observation, InfoPiece

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

import sys

from typing import Optional
from pydantic import BaseModel, Field, field_validator
from answerbot.observation import Observation


class ReflectionResult(BaseModel):
    what_have_we_learned: Optional[str] = Field(..., description="Have we learned anything that would help us answer the user question from the retrieved information and why?")
    comment: str = Field(..., description="A comment on the retrieved information.")
    relevant_quotes: list[str] = Field(..., description="A list of relevant quotes from the source that should be saved.")
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

class KnowledgeBase(BaseModel):
    checked_urls: Optional[dict[str, list[str]]] = Field(default_factory=dict, description="A list of urls that have been checked with learned information.")

    def add_info(self, url: str, info: Optional[str] = None):
        if url not in self.checked_urls:
            self.checked_urls[url] = []
        self.checked_urls[url].append(info)

    def urls(self):
        return self.checked_urls.keys()

    def __str__(self):
        content = ''
        for url, info_list in self.checked_urls.items():
            content += f"\n- {url}"
            for info in info_list:
                content += f"\n - {info}"
            if len(info_list) == 0:
                content += f"\n - no relevant information found"
        return content


import sys

from pydantic import BaseModel, Field
from answerbot.wiki_tool import Observation


class ReflectionResult(BaseModel):
    comment: str = Field(..., description="A comment on the search results and next actions.")
    relevant_quotes: list[str] = Field(..., description="A list of relevant quotes from the source that should be saved.")
    new_sources: list[str] = Field(..., description="A list of new urls mentioned in the notes that should be checked later.")

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
            content += f"Here are quotes that looks relevant:\n{quotes_string}"
        if self.new_sources:
            new_sources_string = "".join("\n - " + link for link in self.new_sources)
            content += f"\nSome links from the notes that might contain relevant information that we should check later:\n{new_sources_string}"
        if len(self.comment) > 0:
            content += f"\n{self.comment}"
        return content
        


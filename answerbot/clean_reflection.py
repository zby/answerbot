import sys

from typing import Optional
from pydantic import BaseModel, Field, field_validator
from answerbot.tools.observation import Observation

from fuzzywuzzy import fuzz

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
        return KnowledgePiece(url=observation.get_current_url(), quotes=checked_quotes, learned=self.what_have_we_learned)

class KnowledgePiece(BaseModel):
    url: str = Field(..., description="The url of the source that provided the information.")
    quotes: list[str] = Field(..., description="A list of quotes from the source.")
    learned: Optional[str] = Field(None, description="The information that was learned from the source.")

    def quoted_quotes(self, indent: Optional[str] = "> "):
        quoted_quotes = []
        for quote in self.quotes:
            quoted_quotes.append(indent + quote.replace("\n", "\n" + indent))
        return '\n\n'.join(quoted_quotes)

    def is_empty(self):
        return self.learned is None and len(self.quotes) == 0

    def __str__(self):
        if self.is_empty():
            return ""
        content = '\n'
        if self.learned is not None:
            text = "\n" + self.learned
            text = text.replace("\n", "\n    ")
            content += f"Learned:{text}\n"
        if len(self.quotes) > 0:
            text = "\n" + self.quoted_quotes()
            text = text.replace("\n", "\n    ")
            content += f"Quotes:{text}\n"
        return content


class KnowledgeBase(BaseModel):
    knowledge_dict: dict[str, list[KnowledgePiece]] = Field(default_factory=dict, description="A dictionary of pieces of knowledge that have been learned.")

    def add_knowledge_piece(self, piece: KnowledgePiece):
        if piece.url not in self.knowledge_dict:
            self.knowledge_dict[piece.url] = []
        if not piece.is_empty():
            self.knowledge_dict[piece.url].append(piece)

    def add_info(self, url: str, quotes: list[str], learned: Optional[str] = None):
        if url not in self.knowledge_dict:
            self.knowledge_dict[url] = []
            if learned is not None or len(quotes) > 0:
                self.knowledge_dict[url].append(KnowledgePiece(url=url, learned=learned, quotes=quotes))
        else:
            if learned is not None or len(quotes) > 0:
                self.knowledge_dict[url].append(KnowledgePiece(url=url, learned=learned, quotes=quotes))

    def urls(self):
        return self.knowledge_dict.keys()

    def __str__(self):
        content = ''
        for url, info_list in self.knowledge_dict.items():
            content += f"\n- {url}"
            for info in info_list:
                text = str(info)
                content += text.replace("\n", "\n    ")
            if len(info_list) == 0:
                content += f"\n       - No relevant information found"
        return content

    def learned(self):
        content = ''
        for url, info_list in self.knowledge_dict.items():
            content += f"\n- {url}, learned:"
            learned = False
            for info in info_list:
                if info.learned is not None:
                    text = "\n" + info.learned
                    text = text.replace("\n", "\n    ")
                    content += f"{text}\n"
                    learned = True
            if not learned:
                content += f"\n    No relevant information found"
        return content

if __name__ == "__main__":
    # Create KnowledgePiece instances and assign them to variables
    knowledge_piece1 = KnowledgePiece(
        url="https://example.com/article1",
        quotes=[
            "This is a significant discovery.\nThe implications are vast", 
            "The new particle that was discovered has implications for many sectors."
        ],
        learned="Discovery of a new particle."
    )

    knowledge_piece2 = KnowledgePiece(
        url="https://example.com/article2",
        quotes=[
            "The policy changes are expected next year.\nThey are not likely to be major.",
            "The new policy will have a significant impact on the economy."
        ],
        learned="Upcoming policy changes in economics."
    )

    knowledge_piece3 = KnowledgePiece(
        url="https://example.com/article3",
        quotes=[
            "Something something.\nSomething something.",
            "Something something.\nSomething something."
        ],
    )

    # Create a sample KnowledgeBase instance
    knowledge_base = KnowledgeBase()

    # Adding the knowledge pieces to the knowledge base
    knowledge_base.add_knowledge_piece(knowledge_piece1)
    knowledge_base.add_knowledge_piece(knowledge_piece2)
    knowledge_base.add_knowledge_piece(knowledge_piece3)

    # Stringify the KnowledgeBase instance to see its content
    print(str(knowledge_base))
    print()
    print('='*100)
    print(knowledge_base.learned())

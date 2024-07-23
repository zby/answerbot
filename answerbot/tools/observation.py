from dataclasses import dataclass, field
from typing import Optional

@dataclass(frozen=True)
class Observation:
    content: str
    operation: Optional[str]
    source: Optional[str] = None
    quotable: bool = False
    goal: Optional[str] = None

    def __str__(self) -> str:
        result = f"**Operation:** {self.operation}\n\n"
        if self.goal:
            result += f"**Goal:** {self.goal}\n\n"
        result += f"{self.content}"
        return result


@dataclass(frozen=True)
class KnowledgePiece:
    """A piece of knowledge extracted from the source observation that can be used to reach the goal and which is supported by quotes."""
    source: Observation
    content: str
    quotes: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not len(self.content) and not len(self.quotes)

    def __str__(self):
        if self.is_empty():
            return ""
        content = '\n'
        if self.content is not None:
            text = "\n" + self.content
            text = text.replace("\n", "\n    ")
            content += f"Learned:{text}\n"
        if len(self.quotes) > 0:
            text = "\n" + self.quoted_quotes()
            text = text.replace("\n", "\n    ")
            content += '\n\n'
            content += f"Quotes:{text}\n"
        return content


@dataclass
class History:
    observations: list[Observation] = field(default_factory=list)
    knowledge_pieces: list[KnowledgePiece] = field(default_factory=list)

    def add_observation(self, observation: Observation):
        self.observations.append(observation)

    def add_knowledge_piece(self, knowledge_piece: KnowledgePiece):
        self.knowledge_pieces.append(knowledge_piece)

    def find_related_knowledge_pieces(self, observation: Observation) -> list[KnowledgePiece]:
        return [kp for kp in self.knowledge_pieces if kp.source == observation]

    def latest_knowledge(self) -> list[KnowledgePiece]:
        if len(self.observations) == 0:
            return []
        return self.find_related_knowledge_pieces(self.observations[-1])

    def sources(self) -> list[str]:
        return [o.source for o in self.observations]
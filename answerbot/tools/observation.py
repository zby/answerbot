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
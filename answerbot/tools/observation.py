from dataclasses import dataclass, field
from typing import Optional

@dataclass
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

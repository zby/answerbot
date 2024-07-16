from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Observation:
    content: str
    operation: Optional[str]
    source: Optional[str] = None
    quotable: bool = False

    def __str__(self) -> str:
        result = f"""**Operation:** {self.operation}


{self.content}"""
        return result

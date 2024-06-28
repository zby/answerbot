from dataclasses import dataclass, field
from typing import Optional

@dataclass
class InfoPiece:
    text: str
    source: Optional[str] = None
    quotable: bool = False
    def to_markdown(self) -> str:
        if self.source and self.quotable:
            return f"{self.text}\n— *from {self.source}*"
        else:
            return self.text

@dataclass
class Observation:
    info_pieces: list[InfoPiece]
    is_refined: bool = False
    interesting_links: list[str] = field(default_factory=list)
    comment: Optional[str] = None
    keyword: Optional[str] = None
    available_tools: str = ''
    current_url: Optional[str] = None
    operation: Optional[str] = None

    def clear_info_pieces(self):
        self.info_pieces = []

    def add_info_piece(self, info_piece: InfoPiece):
        self.info_pieces.append(info_piece)


    def __str__(self) -> str:
        result = f"**Operation:** {self.operation}"
        for info in self.info_pieces:
            result += f"\n\n{info.to_markdown()}"
        if self.interesting_links:
            result += '\n\nNew sources:\n'
            for link in self.interesting_links:
                result += f"\n\n- {link}"
        if self.comment:
            result += f"\n\nComment: {self.comment}"
        return result

    def reflection_needed(self) -> bool:
        return any(info.quotable for info in self.info_pieces)


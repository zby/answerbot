from dataclasses import dataclass, field
from typing import Optional

from answerbot.chat import Prompt, SystemPrompt

from answerbot.tools.observation import Observation
from answerbot.knowledgebase import KnowledgeBase

@dataclass(frozen=True)
class Question(Prompt):
    question: str
    max_llm_calls: int

@dataclass
class Answer:
    """
    Answer to the question.
    """
    answer: str
    reasoning: str

@dataclass(frozen=True)
class StepInfo(Prompt):
    step: int
    max_steps: int

@dataclass(frozen=True)
class PlanningPrompt(Prompt):
    question: str
    available_tools: str
    observation: Optional[Observation] = None
    reflection: Optional[str] = None
    history: list[str] = field(default_factory=list)

@dataclass(frozen=True)
class PlanningSystemPrompt(Prompt):
    pass

@dataclass(frozen=True)
class ReflectionSystemPrompt(Prompt):
    pass

@dataclass(frozen=True)
class ReflectionPrompt(Prompt):
    memory: KnowledgeBase
    question: str
    observation: Observation
    history: list[str] = field(default_factory=list)

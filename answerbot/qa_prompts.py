from dataclasses import dataclass, field
from answerbot.chat import Prompt, Chat
from answerbot.tools.observation import KnowledgePiece, History, Observation
from answerbot.tools.wiki_tool import WikipediaTool

from llm_easy_tools import get_tool_defs

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
    available_tools: list[dict]
    history: History
    new_sources: list[str] = field(default_factory=list)

@dataclass(frozen=True)
class PlanningInsert(Prompt):
    planning_string: str
    knowledge_pieces: list[KnowledgePiece] = field(default_factory=list)
    new_sources: list[str] = field(default_factory=list)

@dataclass(frozen=True)
class PlanningSystemPrompt(Prompt):
    pass

@dataclass(frozen=True)
class ReflectionSystemPrompt(Prompt):
    pass

@dataclass(frozen=True)
class ReflectionPrompt(Prompt):
    history: History
    question: str

def indent_and_quote(text, indent=4, quote_char='>', width=70):
    lines = text.split('\n')
    result = []
    for line in lines:
        result.append(' ' * indent + quote_char + ' ' + line.strip())
    return '\n'.join(result)

if __name__ == '__main__':
    wikipedia_tool = WikipediaTool()
    tools = wikipedia_tool.get_llm_tools()
    schemas = get_tool_defs(tools)

    templates_dirs = ['answerbot/templates/common/', 'answerbot/templates/wiki_researcher/'] 
    chat = Chat(
        model='gpt-4o',
        templates_dirs=templates_dirs,
    )
    chat.template_env.filters['indent_and_quote'] = indent_and_quote
    
    observation = Observation("France is a country in Europe", "research")

    for prompt in [
        Question('What is the capital of France?', 3),
        Answer('Paris', 'According to sources'),
        StepInfo(1, 3),
        PlanningPrompt('What is the capital of France?', schemas, History(), []),
        PlanningInsert('I would do search', [KnowledgePiece(observation, "France is a country", ["France is a country in Europe"])], ['https://www.google.com']),
        PlanningSystemPrompt(),
        ReflectionSystemPrompt(),
        ReflectionPrompt(History(), 'What is the capital of France?')
        ]:
        print(f"\n\n\nprompt: {prompt.__class__.__name__}\n")
        print(chat.render_prompt(prompt))

    history = History()
    observation2 = Observation("Germany is a country in Europe", "research")
    history.add_observation(observation)
    history.add_observation(observation2)
    print(chat.render_prompt(PlanningPrompt('What is the capital of France?', schemas, history, [])))
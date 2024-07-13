from dataclasses import dataclass, field
from typing import Callable, Any, Optional

from llm_easy_tools import LLMFunction, get_tool_defs

import logging

from answerbot.chat import Chat, HasLLMTools, SystemPrompt, expand_toolbox
from answerbot.tools.observation import Observation, InfoPiece
from answerbot.reflection_result import ReflectionResult 
from answerbot.knowledgebase import KnowledgeBase, KnowledgePiece
from answerbot.qa_prompts import Question, Answer, StepInfo, ReflectionPrompt, ReflectionSystemPrompt, PlanningPrompt, PlanningSystemPrompt

# Configure logging for this module
logger = logging.getLogger('qa_processor')

def format_tool_docstrings(schemas: list[dict]) -> str:
    formatted_list = []
    for schema in schemas:
        func_name = schema['function']['name']
        description = schema['function'].get('description', '')

        # Start with function name and description
        doc = f"- **{func_name}**\n\n"
        doc += "\n".join(f"  {line}" for line in description.split('\n')) + "\n\n"

        # Add parameters section if present
        if 'parameters' in schema['function']:
            doc += "  Parameters\n  ----------\n"
            properties = schema['function']['parameters'].get('properties', {})
            for param, details in properties.items():
                param_type = details.get('type', 'Any')
                param_desc = details.get('description', '')
                doc += f"  {param} : {param_type}\n"
                doc += "\n".join(f"      {line}" for line in param_desc.split('\n')) + "\n"

        formatted_list.append(doc)

    return "\n".join(formatted_list)


@dataclass(frozen=True)
class QAProcessor:
    toolbox: list[HasLLMTools|LLMFunction|Callable]
    max_iterations: int
    model: str
    prompt_templates: dict[str, str] = field(default_factory=dict)
    prompt_templates_dirs: list[str] = field(default_factory=list)
    name: Optional[str] = None

    def get_tools(self, step: int) -> list[Callable|LLMFunction|HasLLMTools]:
        if step < self.max_iterations:
            return[Answer, *self.toolbox]
        else:
            return [Answer]

    def make_chat(self) -> Chat:
        chat = Chat(
            model=self.model,
            one_tool_per_step=True,
            templates=self.prompt_templates,
            templates_dirs=self.prompt_templates_dirs,
            context=self,
        )
        if self.name:
            chat.metadata = {"tags": [self.name]}
        return chat

    def process(self, question: str):
        logger.info(f'Processing question: {question}')
        chat = self.make_chat()
        chat.append(SystemPrompt())
        chat.append(Question(question, self.max_iterations))

        what_have_we_learned = KnowledgeBase()

        observation = None
        reflection_string = None
        for step in range(self.max_iterations + 1):
            chat.append(StepInfo(step, self.max_iterations))
            tools = self.get_tools(step)
            results = chat.process(tools)
            if not results:
                logger.warn("No tool call in a tool loop")
            else:
                output = results[0]
                if isinstance(output, Answer):
                    answer = output
                    logger.info(f"Answer: '{answer}' for question: '{question}'")
                    return chat.render_prompt(answer, {'question': question})
                observation = output
                if isinstance(output, Observation):
                    if observation.reflection_needed():
                        reflection_string = self.reflect(question, observation, what_have_we_learned)
                planning_string = self.plan_next_action(question, observation, reflection_string)
                chat.append({'role': 'user', 'content': planning_string})
            logger.info(f"Step: {step} for question: '{question}'")

        return None

    def reflect(self, question: str, observation: Observation, knowledge_base: KnowledgeBase) -> str:
        chat = self.make_chat()
        chat.append(ReflectionSystemPrompt())
        chat.append(ReflectionPrompt(memory=knowledge_base, question=question, observation=observation))

        reflections = []
        for reflection in chat.process([ReflectionResult]):
            if observation.current_url:
                reflections.append(reflection.update_knowledge_base(knowledge_base, observation))
        reflection_string = '\n'.join(reflections)

        return reflection_string

    def plan_next_action(self, question: str, observation: Optional[Observation] = None, reflection_string: Optional[str] = None) -> str:
        chat = self.make_chat()
        chat.append(PlanningSystemPrompt())

        schemas = get_tool_defs(expand_toolbox(self.get_tools(0)))
        available_tools_str = format_tool_docstrings(schemas)

        planning_prompt = PlanningPrompt(
            question=question,
            available_tools=available_tools_str,
            observation=observation,
            reflection=reflection_string
        )

        chat.append(planning_prompt)
        response = chat.llm_reply()
        planning_result = response.choices[0].message.content

        planning_string = f"**My Notes**\n{reflection_string}\n\nHmm what I could do next?\n\n{planning_result}"

        return planning_string



@dataclass(frozen=True)
class QAProcessorDeep(QAProcessor):
    sub_processor_config: Optional[dict[str, Any]] = None
    delegate_description: Optional[str] = None

    def __post_init__(self):
        if self.sub_processor_config:
            if self.toolbox:
                raise(Exception("Cannot set toolbox when using sub processor"))
            if self.delegate_description is None:
                raise(Exception("Must set deletage_description when usign sub processor"))

            sub_processor =  QAProcessor(**self.sub_processor_config)

            def delegate(sub_question: str):
                logger.info(f"{self.delegate_description}: '{sub_question}'")
                return sub_processor.process(sub_question)

            delegate_fun = LLMFunction(delegate, description=self.delegate_description)

            object.__setattr__(self, 'toolbox', [delegate_fun])

if __name__ == "__main__":
    from dotenv import load_dotenv
    import litellm

    load_dotenv()
    litellm.success_callback = ["langfuse"]
    litellm.failure_callback = ["langfuse"]

    # Configure logging for the chat module
    #chat_logger = logging.getLogger('answerbot.chat')
    #chat_logger.setLevel(logging.DEBUG)
    #chat_logger.addHandler(logging.StreamHandler(sys.stdout))


    qa_processor = QAProcessor(
        toolbox=[],
        max_iterations=1,
        model='gpt-3.5-turbo',
        prompt_templates_dirs=['answerbot/templates/wiki_researcher/'],
        name='test'
    )

    # Create an example Observation
    observation = Observation(
        current_url="https://en.wikipedia.org/wiki/Byzantine_Empire",
        info_pieces=[
            InfoPiece("Constantinople was the capital of the Byzantine Empire.", source="Wikipedia", quotable=True),
            InfoPiece("The Byzantine Empire lasted from 330 AD to 1453 AD.", source="History.com", quotable=True)
        ],
        operation="Initial search"
    )

    # Example knowledge base
    what_have_we_learned = KnowledgeBase()
    new_knowledge_piece = KnowledgePiece(
        url="https://www.britannica.com/place/Byzantine-Empire",
        quotes=["The Byzantine Empire was also known as the Eastern Roman Empire and was a continuation of the Roman Empire in its eastern provinces."],
        learned="The Byzantine Empire was a continuation of the Roman Empire in its eastern provinces and was also known as the Eastern Roman Empire."
    )

    # Add the new KnowledgePiece to the knowledge base
    what_have_we_learned.add_knowledge_piece(new_knowledge_piece)

    question = 'What is the capital of the Eastern Roman Empire?'

    reflection_string = qa_processor.reflect(question, observation, what_have_we_learned)

    print()
    print(reflection_string)
    print()
    print(qa_processor.plan_next_action(question, observation, reflection_string))
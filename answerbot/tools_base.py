from llm_easy_tools import llm_function
from pydantic import BaseModel, Field


@llm_function()
class Finish(BaseModel):
    """
    Finish the task and return the answer.
    """
    answer: str = Field(description="The answer to the user's question")
    answer_short: str = Field(description="A short version of the answer")
    reasoning: str = Field(description="The reasoning behind the answer. Think step by step. Mention all assumptions you make.")
    ambiguity: str|None = Field(description="Have you found anything in the retrieved information that makes the question ambiguous? For example a search for some name can show that there are many different entities with the same name.")

    def normalize_answer(self, answer):
        answer = answer.strip(' \n.\'"')
        answer = answer.replace('’', "'")  # Replace all curly apostrophes with straight single quotes
        answer = answer.replace('"', "'")  # Replace all double quotes with straight single quotes
        if answer.lower() == 'yes' or answer.lower() == 'no':
            answer = answer.lower()
        return answer

    def normalized_answer(self):
        return (
            self.normalize_answer(self.answer),
            self.normalize_answer(self.answer_short),
        )


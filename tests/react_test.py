from pydantic import BaseModel, Field

from answerbot.react import Reflection

class ShortReflection(BaseModel):
    reflection: str = Field(..., description="Reflect on the information you have gathered so far. Was the last retrieved information relevant for answering the question? What additional information you need, why and how you can get it? Think step by step")


def test_reflection():
    reflection = Reflection(reflection_class=ShortReflection, detached=False, case_insensitive=True)
    assert reflection.prefix() == "shortreflection_and_"
    reflection = Reflection(reflection_class=ShortReflection, detached=False, case_insensitive=False)
    assert reflection.prefix() == "ShortReflection_and_"


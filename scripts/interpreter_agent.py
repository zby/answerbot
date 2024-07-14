import io
import sys
import contextlib

from dotenv import load_dotenv

from answerbot.qa_processor import QAProcessor
from answerbot.tools.observation import Observation, InfoPiece

load_dotenv()

# Set logging level to INFO for qa_processor.py
#logging.getLogger('qa_processor').setLevel(logging.INFO)
import logging
import litellm
qa_logger = logging.getLogger('qa_processor')
qa_logger.setLevel(logging.INFO)
chat_logger = logging.getLogger('answerbot.chat')
chat_logger.setLevel(logging.DEBUG)

# Create a console handler
console_handler = logging.StreamHandler()

# Add the handler to the logger
qa_logger.addHandler(console_handler)
chat_logger.addHandler(console_handler)

litellm.success_callback = ["langfuse"]
litellm.failure_callback = ["langfuse"]


def execute(code: str) -> str:
    """Run python code and return the captured stdout and stderr. Don't forget to use a print statement to see the results."""
    
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
        try:
            exec(code)
        except Exception as e:
            print(f"An error occurred: {str(e)}", file=sys.stderr)

    output = stdout_capture.getvalue().strip()
    error = stderr_capture.getvalue().strip()

    if error:
        output += '\n\n' + error
    operation = f'executing: \n"""\n{code}\n"""'

    info = InfoPiece(text=output, source="interpreter")
    observation = Observation(info_pieces=[info], operation=operation)

    return observation

prompts = {
    'SystemPrompt': "You are an expert Python programmer. You answer questions by running Python code.",
    'Question': "Question: {{question}}",
    'StepInfo': "Step: {{step + 1}} of {{max_steps + 1}}",
    'Answer': "Answer: {{answer}}\n\nReasoning: {{reasoning}}",
    'PlanningPrompt': """
# Question

The user's question is: {{question}}

# Available tools

{{available_tools}}

{% if observation %}
# Observation

{{observation}}
{% endif %}

# Next step

What would you do next?
For now specify only the next step. Use Markdown syntax.
Explain your reasoning.

Please be precise and specify both the tool together with the parameters you need as a function call, something like `function(parameter)`.
"""
}

if __name__ == "__main__":

    app = QAProcessor(
        toolbox=[execute],
        max_iterations=5,
        model='gpt-3.5-turbo',
        prompt_templates=prompts,
        fail_on_tool_error=False
    )

    question = "What is sin(33.3)?"

    print(app.process(question))
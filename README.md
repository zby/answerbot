# answerbot

## Installation

Copy `.env_` to `.env` and fill in the blanks.

Install https://github.com/zby/ToolDefGenerator

Install the requirements with `pip install -r requirements.txt`

Install the package. I am installing it in an editable form: `pip install -e .`

Run tests:
`pytest tests/`


## Scripts

### `scripts/answer.py`
Runs a question and answer session.
Edit the file to change the question.

### `scripts/run_experiment.py`
It runs an experiment on a full product of sets of available values for config options and available questions.
This might be a very big set!

Edit the file to choose config options to be evaluated.

It can run questions from a file or you can just edit the list of questions at the start of the script.

Results are saved in a new subdir in the `experiments` directory.

We have our own set of questions, they are rather tricky but you can run it with them:

`python scripts/run_experiment.py data/questions.json`

More questions in in the same format in:
[https://github.com/andyz245/LanguageAgentTreeSearch/blob/main/hotpot/data/](https://github.com/andyz245/LanguageAgentTreeSearch/blob/main/hotpot/data/)

### Example code in `if __name__ == "__main__"` blocks
- `answerbot/get_wikipedia.py`
- `answerbot/document.py`

## Compatibility
I currently work only on OpenAI llms.

In particular I use function calls. I have some code for parsing llm answers and retrieve function calls from it - but I never used it.

## Generating examples
The current 0-shot prompts seem to work - so I have not tested the example generating code for long time.

## TODO
* ongoing refactoring for easier use in Jupyter notebooks
* `answerbot/react_prompt.py` contains a lot of unused code
* working on less hallucinations in summaries
* stats in experiments
* better prompt testing in experiments
* better prompts
* detecting loops
* following links

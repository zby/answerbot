# answerbot

## Installation

Copy `config.json_` to `config.json` and fill in the blanks.

Install the requirements with `pip install -r requirements.txt`

Install the library package. I am installing it in an editable form: `pip install -e .`

Run tests:
`python -m unittest discover -p '*_test.py' -s tests/`


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

## TODO
* ongoing refactoring for easier use in Jupyter notebooks
* working on less hallucinations in summaries
* stats in experiments
* better prompt testing in experiments
* better prompts
* detecting loops

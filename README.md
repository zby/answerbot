# answerbot

## Scripts

### `react.py`
This is mostly a library - but if it is executed it runs one example.
Edit the end of the file to change the example.

### `run_experiment.py`
It runs an experiment on a full product of sets of available values for config options and available questions.
This might be a very big set!

Edit the file to choose config options to be evaluated.

It can run questions from a file or you can just edit the list of questions at the start of the script.

Results are saved in a new subdir in the `experiments` directory.

We have our own set of questions, they are rather tricky but you can run it with them:

`python scripts/run_experiment.py data/questions.json`

More questions in in the same format in:
[https://github.com/andyz245/LanguageAgentTreeSearch/blob/main/hotpot/data/](https://github.com/andyz245/LanguageAgentTreeSearch/blob/main/hotpot/data/)

## Installation

Copy `config.json_` to `config.json` and fill in the blanks.

Install the requirements with `pip install -r requirements.txt`

Run tests:
`python -m unittest discover -p '*_test.py' -s tests/`

## TODO
* stats in experiments
* better prompt testing in experiments
* better prompts
* detecting loops

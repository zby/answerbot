# answerbot

## Scripts

### `react.py`
This is mostly a library - but if it is executed it runs one example.
Edit the end of the file to change the example.

### `scripts/batch_run.py`
It runs questions from a file given as a parameter.
We have our own set of questions, they are rather tricky but you can run it with them:

`python batch_run.py data/questions.json`

Some more questions are in

`https://github.com/andyz245/LanguageAgentTreeSearch/blob/main/hotpot/data/`

The results are saved in the `logs` directory.

### `scripts/marginal_questions_experiment.py`
This is for checking questions that are on the margin - sometimes are answered well sometimes not.
It runs an experiment on a full product of sets of available values for config options and available questions.
This might be a very big set!
Results are saved in a new subdir in the `experiments` directory.

## Installation

Copy `config.json_` to `config.json` and fill in the blanks.

Install the requirements with `pip install -r requirements.txt`

Run tests:
`python -m unittest discover -p '*_test.py' -s tests/`

## TODO
* something to make the model answers shorter - so that we can compare them to the gold answers (improved a bit recently)
* setting up experiments - with better logging, stats and error recovery so that we can test prompts and constants in various combinations (some initial work done)
* better prompts (maybe negative examples for discouraging it from searching for 'something property' instead first searching for 'something' and then lookup up its property. At wikipedia the first search gives bad results.
* detecting loops

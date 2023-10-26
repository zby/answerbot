# answerbot

Currently working on react.py

batch_run.py also works more or less - as input it takes a file like: 
https://github.com/andyz245/LanguageAgentTreeSearch/blob/main/hotpot/data/hotpot_dev_v1_simplified.json

You can run it with our rather tricky questions:

`python batch_run.py data/questions.json`

## Installation

Copy `config.json_` to `config.json` and fill in the blanks.

Install the requirements with `pip install -r requirements.txt`

Run tests:
`python -m unittest discover -p '*_test.py' -s tests/`

## TODO
* something to make the model answers shorter - so that we can compare them to the gold answers.
* setting up experiments - with better logging, stats and error recovery so that we can test prompts and constants in various combinations.
* better prompts
* detecting loops

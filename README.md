# answerbot
Agentic RAG system for question answering.

Work in progress.

## Installation

Copy `.env_` to `.env` and fill in your OPENAI_API_KEY. You can also set the environment variable in the shell.

Install the requirements with `pip install -r requirements.txt`

Install the package. I am installing it in an editable form: `pip install -e .`

Run tests:
`pytest tests/`


## Scripts

### `scripts/answer.py`
Runs a question and answer session.
Edit the file to change the question.


## Compatibility
I currently work only on OpenAI llms.

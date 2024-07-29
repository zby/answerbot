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
Uses LiteLLM. I use it with OpenAI GPTs and Anthropic Claude. Llama 3 via Groq had too many errors around tool use - I need to test 3.1 and/or other providers of maybe finetuned models.

## Mutable.ai Auto Wiki
Thanks to Mutable.ai auto wiki we have an auto generated wiki for this repo: https://wiki.mutable.ai/zby/answerbot

[![Mutable.ai Auto Wiki](https://img.shields.io/badge/Auto_Wiki-Mutable.ai-blue)](https://wiki.mutable.ai/FollettSchoolSolutions/perfmon4j)
from answerbot.get_wikipedia import WikipediaApi, ContentRecord

wiki_api = WikipediaApi(max_retries=3)

# Example 1: Retrieve a Wikipedia page and its retrieval history
title = "Ukrainian War"
content_record = wiki_api.search(title)
print(content_record.retrieval_history)
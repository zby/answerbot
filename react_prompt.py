import json

import tiktoken
import os

from prompt_builder import FunctionalPrompt, PlainTextPrompt, User, System, Assistant, FunctionCall, FunctionResult
from get_wikipedia import WikipediaDocument, ContentRecord

class Question(User):
    def plaintext(self) -> str:
        return '\nQuestion: ' + self.content
    def openai_message(self) -> dict:
        return { "role": "user", "content": 'Question: ' + self.content }

new_functional_system_message = System('''You are an expert Wikipedia editor. 
Solve a question answering task by interacting with the Wikipedia API.
Please make the answer as short as possible. If it can be answered with yes or no that is best.
Remove all explanations from the answer and put them into a separate field.
When searching for something you need to think what wikipedia page could contain that information. For example
if you want to know the nationality of a person search for the person's name. If you want to know something
about a place or institution search for the place or institution name.
After you get new information you need to reflect on it, check if you got all the needed information
and decide what to do next.
If you found the right page is retrieved, but you don't see the needed information then call the lookup function.
If you lookup for a phrase and you don't find it - then try looking up words from the phrase.
If the page is wrong - then try another search.
For example if you want to know the elevation of 'High Plains' search('High Plains') and if you have the
the right page then lookup('elevation').
The words in double square brackets are links - you can follow them with the get function.
Here are some examples:''')

preamble = '''Solve a question answering task with interleaving Thought, Action, Observation steps.
Please make the answer as short as possible. If it can be answered with a single word, that is best.
Don't put any explanations in the answer, that is what the Thought step is for.
After each Observation you need to reflect on the response in a Thought step.
Thought can reason about the current situation, and Action means looking up more information or finishing the task.
If you want to learn about a property of something first search('something'),
check if the right page is retrieved, check if the information you are looking for is in the results.
If the right page is retrieved, but you don't have the needed information lookup('property').
If the page is wrong - then try another search.
For example if you want to know the elevation of 'High Plains' search('High Plains')
and then lookup('elevation').
'''
plain_system_message = System(
    preamble + '''The available Actions are:
(1) search[query] searches Wikipedia saves the first result page and informs about the content of that page.
(2) lookup[keyword] returns text surrounding the keyword in the current page.
(2) get[title] gets the Wikipedia page with the given title, saves it and informs about the content of that page.
(3) finish[answer] returns the answer and finishes the task.
After each observation, provide the next Thought and next Action. Here are some examples:''')
functional_system_message = System(
    preamble + '''
For the Action step you can call the available functions.
The words in double square brackets are links - you can follow them with the get function.
Here are some examples:''')

def retrieval_observations(search_record, limit_sections = 10):
    observations = ""
    document = search_record.document
    for record in search_record.retrieval_history:
        observations = observations + record + "\n"
    if document is None:
        observations = observations + "No wikipedia page found"
    else:
        sections = document.section_titles()
        sections_list_md = "\n".join(map(lambda section: f' - {section}', sections))
        if limit_sections is not None:
            sections_list_md = sections_list_md[:limit_sections]
        observations = observations + f'The retrieved page contains the following sections:\n{sections_list_md}\n'
        observations = observations + "The retrieved page summary starts with: " + document.first_chunk() + "\n"
    return observations

def lookup_observations(document, keyword):
    if document is None:
        observations = "No document defined, cannot lookup"
    else:
        text = document.lookup(keyword)
        observations = 'Keyword "' + keyword + '" '
        if text:
            observations = observations + "found  in: \n" + text
        else:
            observations = observations + "not found in current page"
    return observations


class ReactPrompt:
    def __init__(self, question, initial_system_message, examples_chunk_size=300):
        self.examples_chunk_size = examples_chunk_size
        self.question = question
        self.initial_system_message = initial_system_message
        examples = self.get_examples()
        super().__init__([self.initial_system_message, *examples, Question(self.question)])

    def mk_record(self, title):
        """
        Load a ContentRecord from saved wikitext and retrieval history files based on a given title.

        Returns:
        - ContentRecord: A ContentRecord object reconstructed from the saved files.
        """
        directory = "data/wikipedia_pages"
        sanitized_title = title.replace("/", "_").replace("\\", "_")  # To ensure safe filenames
        sanitized_title = sanitized_title.replace(" ", "_")
        wikitext_filename = os.path.join(directory, f"{sanitized_title}.txt")
        history_filename = os.path.join(directory, f"{sanitized_title}.retrieval_history")

        # Load wikitext content
        with open(wikitext_filename, "r", encoding="utf-8") as f:
            document_content = f.read()

        # Load retrieval history
        retrieval_history = []
        with open(history_filename, "r", encoding="utf-8") as f:
            for line in f:
                retrieval_history.append(line.strip())

        document = WikipediaDocument(
            document_content, chunk_size=self.examples_chunk_size)
        return ContentRecord(document, retrieval_history)

    def get_examples(self):

        colorado_orogeny_record = self.mk_record(
            'Colorado orogeny',
        )

        high_plains_record = self.mk_record(
            'High Plains',
        )

        high_plains_us_record = self.mk_record(
            'High Plains geology',
        )

        milhouse_record = self.mk_record(
            'Milhouse Van Houten',
        )

        poland_record = self.mk_record(
            'Poland',
        )

        additional_messages = []
        document = high_plains_us_record.document
        first_chunk = document.first_chunk()
        if not 'elevation' in first_chunk:
            additional_messages = [
                FunctionCall(
                    'lookup',
                    keyword='elevation',
                    thought='This passge does not mention elevation. I need to find out the elevation range of the High Plains.'
                ),
                FunctionResult('lookup', lookup_observations(document, 'elevation'))
            ]

        examples = [
            Question("When Poland became elective-monarchy?"),
            FunctionCall(
                'search',
                thought="I need to read about Poland's history to find out when Poland became an elective-monarchy.",
                query="Poland",
            ),
            FunctionResult('search', retrieval_observations(poland_record, 2)),
            FunctionCall(
                'lookup',
                thought='This is the right page. I will lookup "elective-monarchy" here.',
                keyword="elective",
            ),
            FunctionResult('lookup', lookup_observations(poland_record.document, "elective-monarchy")),
            FunctionCall(
                'lookup',
                thought='Hmm. Maybe I will lookup "elective" here.',
                keyword="elective",
            ),
            FunctionResult('lookup', lookup_observations(poland_record.document, "elective")),
            FunctionCall(
                'finish',
                thought="The Union of Lublin of 1569 established the Polish Lithuanian Commonwealth which was an elective monarchy.",
                answer="1569",
            ),
            Question(
                "What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?"),
            FunctionCall(
                "search",
                thought='I need to search Colorado orogeny, find the area that the eastern sector of the Colorado orogeny extends into, then find the elevation range of the area.',
                query="Colorado orogeny",
            ),
            FunctionResult('search', retrieval_observations(colorado_orogeny_record, 2)),
            FunctionCall(
                'lookup',
                thought="This is the right page - but it does not mention the eastern sector of the Colorado orogeny. I need to look up eastern sector.",
                keyword="eastern",
            ),
            FunctionResult('lookup', lookup_observations(colorado_orogeny_record.document, "eastern")),
            FunctionCall(
                'search',
                thought="The eastern sector of Colorado orogeny extends into the High Plains, so High Plains is the area. I need to find out the elevation of High Plains.",
                query="High Plains",
            ),
            FunctionResult('search', retrieval_observations(high_plains_record, 2)),
            FunctionCall(
                'search',
                thought='High Plains Drifter is a film. I am not on the right page I need information about High Plains in geology or geography',
                query="High Plains geology",
            ),
            FunctionResult('search', retrieval_observations(high_plains_us_record, 2)),
            *additional_messages,
            FunctionCall(
                'finish',
                thought='The High Plains have an elevation range from around 1,800 to 7,000 feet. I can use this information to answer the question about the elevation range of the area that the eastern sector of the Colorado orogeny extends into.',
                answer="approximately 1,800 to 7,000 feet",
            ),

            Question(
                'Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?'),
            FunctionCall(
                'search',
                thought='I need to find out who Matt Groening named the Simpsons character Milhouse after.',
                query="Milhouse Simpson",
            ),
            FunctionResult('search', retrieval_observations(milhouse_record, 2)),
            FunctionCall(
                'lookup',
                thought='This is the right page - but the summary does not tell who Milhouse is named after, the section called "Creation" should contain information about how it was created and maybe who Milhouse was named after.',
                keyword="Creation",
            ),
            FunctionResult('lookup', lookup_observations(milhouse_record.document, "Creation")),
            FunctionCall(
                'finish',
                thought="Milhouse was named after U.S. president Richard Nixon, so the answer is President Richard Nixon.",
                answer="President Richard Nixon",
            ),

        ]

        return examples

class FunctionalReactPrompt(ReactPrompt, FunctionalPrompt):
    def __init__(self, question, examples_chunk_size):
        super().__init__(question, functional_system_message, examples_chunk_size)

    def function_call_from_response(self, response):
        return response.get("function_call")

class NewFunctionalReactPrompt(ReactPrompt, FunctionalPrompt):
    def __init__(self, question, examples_chunk_size):
        super().__init__(question, new_functional_system_message, examples_chunk_size)

    def function_call_from_response(self, response):
        return response.get("function_call")


class TextReactPrompt(ReactPrompt, PlainTextPrompt):
    def __init__(self, question, examples_chunk_size):
        super().__init__(question, plain_system_message, examples_chunk_size)

    def function_call_from_response(self, response):
        response_lines = response["content"].strip().split('\n')
        if len(response_lines) >= 2:
            last_but_one_line = response_lines[-2]
        else:
            last_but_one_line = ""
        last_line = response_lines[-1]
        if last_line.startswith('Action: '):
            if last_but_one_line.startswith('Thought: '):
                thought = last_but_one_line[9:]
            else:
                thought = None
            if last_line.startswith('Action: finish['):
                answer = last_line[15:-1]
                return {
                    "name": "finish",
                    "arguments": json.dumps({"answer": answer, "thought": thought})
                }
            elif last_line.startswith('Action: search['):
                query = last_line[15:-1]
                return {
                    "name": "search",
                    "arguments": json.dumps({"query": query, "thought": thought})
                }
            elif last_line.startswith('Action: get['):
                query = last_line[15:-1]
                return {
                    "name": "get",
                    "arguments": json.dumps({"title": query, "thought": thought})
                }
            elif last_line.startswith('Action: lookup['):
                keyword = last_line[15:-1]
                return {
                    "name": "lookup",
                    "arguments": json.dumps({"keyword": keyword, "thought": thought})
                }
        return None

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

if __name__ == "__main__":

#    frprompt = FunctionalReactPrompt("Bla bla bla", 300)
#    erecord = frprompt.mk_record('Poland')
#    search_res = FunctionResult('search', retrieval_observations(erecord))
#    print(search_res.plaintext())
#    lookup_res = FunctionResult('lookup', lookup_observations(erecord.document, "elective monarchy"))
#    print(lookup_res.plaintext())


    frprompt = FunctionalReactPrompt("Bla bla bla", 300)
    trprompt = TextReactPrompt("Bla bla bla", 300)

    print(frprompt.to_text())
    print()
    print("-" * 80)
    print()
    print(trprompt.to_text())
    # print a line separator
    print()
    print("-" * 80)
    print()
    print("The length of the text prompt is: " + str(len(trprompt.to_text())) + " characters.")
    print("The length of the text prompt is: " + str(num_tokens_from_string(trprompt.to_text(), "cl100k_base")) + " tokens.")

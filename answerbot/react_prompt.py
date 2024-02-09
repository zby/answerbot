import json

import tiktoken

from .prompt_builder import FunctionalPrompt, PlainTextPrompt, User, System, FunctionCall, FunctionResult
from .prompt_templates import Question
from .get_wikipedia import WikipediaApi
from .wikipedia_tool import WikipediaSearch


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


class ReactPrompt:
    def __init__(self, question, initial_system_message, examples_chunk_size=300):
        self.examples_chunk_size = examples_chunk_size
        self.question = question
        self.initial_system_message = initial_system_message
        wiki_api = WikipediaApi(max_retries=3, chunk_size=examples_chunk_size)
        self.toolbox = WikipediaSearch(wiki_api, cached=True)
        examples = self.get_examples()
        super().__init__([self.initial_system_message, *examples, Question(self.question)])


    def mk_example_call(self, name, **args):
        fcall = FunctionCall(name, **args)
        if name == 'finish':
            return [fcall]
        method = self.toolbox._method_from_tool(name)
        result = method(**args)
        return [fcall, FunctionResult(name, result)]

    def mk_additional_lookup_if_needed(self, keyword):
        if not keyword in self.toolbox.document.first_chunk():
            name = 'lookup'
            args = {"keyword": keyword, "reason": f'This is the right page - but it does not mention "f{keyword}". I need to look up "{keyword}".'}
            return self.mk_example_call(name, **args)
        else:
            return []

    def get_examples(self):
        examples = []
        examples.append(Question("When Poland became elective-monarchy?"))
        examples.extend(self.mk_example_call("search", query="Poland", reason="I need to read about Poland's history to find out when Poland became an elective-monarchy."))
        examples.extend(self.mk_example_call("lookup", keyword="elective-monarchy", reason="This is the right page. I will lookup \"elective-monarchy\" here."))
        examples.extend(self.mk_example_call("lookup", keyword="elective", reason="Hmm. Maybe I will lookup \"elective\" here."))
        examples.extend(self.mk_example_call("finish", answer="1569", reason="The Union of Lublin of 1569 established the Polish Lithuanian Commonwealth which was an elective monarchy."))

        examples.append(Question("What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?"))
        examples.extend(self.mk_example_call("search", query="Colorado orogeny", reason="I need to search Colorado orogeny, find the area that the eastern sector of the Colorado orogeny extends into, then find the elevation range of the area."))
        examples.extend(self.mk_example_call("lookup", keyword="eastern", reason="This is the right page - but it does not mention the eastern sector of the Colorado orogeny. I need to look up eastern sector."))
        examples.extend(self.mk_example_call("search", query="High Plains", reason="The eastern sector of Colorado orogeny extends into the High Plains, so High Plains is the area. I need to find out the elevation of High Plains."))
        examples.extend(self.mk_example_call("search", query="High Plains geology", reason="High Plains Drifter is a film. I am not on the right page I need information about High Plains in geology or geography"))
        examples.extend(self.mk_additional_lookup_if_needed("elevation"))
        examples.extend(self.mk_example_call("finish", answer="approximately 1,800 to 7,000 feet", reason="The High Plains have an elevation range from around 1,800 to 7,000 feet. I can use this information to answer the question about the elevation range of the area that the eastern sector of the Colorado orogeny extends into."))

        examples.append(Question(
            'Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?'))
        examples.extend(self.mk_example_call("search", query="Milhouse Simpsons", reason="I need to find out who Matt Groening named the Simpsons character Milhouse after."))
        examples.extend(self.mk_example_call("lookup", keyword="named after", reason="This is the right page - but the summary does not tell who Milhouse is named after, I'll lookup 'named after' here."))
        examples.extend(self.mk_example_call("finish", answer="President Richard Nixon", reason="Milhouse was named after U.S. president Richard Nixon, so the answer is President Richard Nixon."))

        return examples

class FunctionalReactPrompt(ReactPrompt, FunctionalPrompt):
    def __init__(self, question, examples_chunk_size):
        super().__init__(question, functional_system_message, examples_chunk_size)


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
                reason = last_but_one_line[9:]
            else:
                reason = None
            if last_line.startswith('Action: finish['):
                answer = last_line[15:-1]
                return {
                    "name": "finish",
                    "arguments": json.dumps({"answer": answer, "reason": reason})
                }
            elif last_line.startswith('Action: search['):
                query = last_line[15:-1]
                return {
                    "name": "search",
                    "arguments": json.dumps({"query": query, "reason": reason})
                }
            elif last_line.startswith('Action: get['):
                query = last_line[15:-1]
                return {
                    "name": "get",
                    "arguments": json.dumps({"title": query, "reason": reason})
                }
            elif last_line.startswith('Action: lookup['):
                keyword = last_line[15:-1]
                return {
                    "name": "lookup",
                    "arguments": json.dumps({"keyword": keyword, "reason": reason})
                }
        return None

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

if __name__ == "__main__":

    #toolbox = WikipediaSearch(WikipediaApi(max_retries=3, chunk_size=300))
    #print(toolbox.process("search", {"query": "Guns N Roses", "thought": "I need to read about Guns N Roses."}))

    frprompt = FunctionalReactPrompt("Bla bla bla", 200)
    # _, result = frprompt.mk_example_call("search", query="Milhouse Simpsons", reason="I need to find out who Matt Groening named the Simpsons character Milhouse after.")
    # _, result = frprompt.mk_example_call("lookup", keyword="named after", reason="This is the right page - but the summary does not tell who Milhouse is named after, I'll lookup 'named after' here.")
    # print(result.openai_message()['content'])
    # exit()

    trprompt = TextReactPrompt("Bla bla bla", 200)

    print(frprompt)
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

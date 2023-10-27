import tiktoken
import os

from prompt_builder import Prompt, PromptMessage, User, InitialSystemMessage, Assistant, FunctionCall, FunctionResult
from get_wikipedia import WikipediaDocument, ContentRecord

EXAMPLES_CHUNK_SIZE = 300


class Question(User):
    def plaintext(self) -> str:
        return '\nQuestion: ' + self.content
    def openai_message(self) -> dict:
        return { "role": "user", "content": 'Question: ' + self.content }



preamble = '''You are a helpful AI assistant trying to answer questions.
Please make the answer short and concise.
Solve a question answering task with interleaving Thought, Action, Observation steps. 
After each Observation you need to reflect on the response in a Thought step.
Thought can reason about the current situation, and Action means looking up more information or finishing the task.'''
system_message = InitialSystemMessage(
    preamble + '''
(1) search[query], which searches Wikipedia saves the first result page and informs about the content of that page.
(2) lookup[keyword], which returns text surrounding the keyword in the current page.
(2) get[title], which gets the Wikipedia page with the given title, saves it and informs about the content of that page.
(3) finish[answer], which returns the answer and finishes the task.
After each observation, provide the next Thought and next Action. Here are some examples: 

''',
    preamble + '''
For the Action step you can call the available functions.
The words in double square brackets are links - you can follow them with the get function.
Here are some examples:

''',
)
def retrieval_observations(search_record):
    observations = ""
    document = search_record.document
    for record in search_record.retrieval_history:
        observations = observations + record + "\n"
    if document is None:
        observations = observations + "No wikipedia page found"
    else:
        observations = observations + "The retrieved wikipedia page summary starts with: " + document.first_chunk() + "\n"

        sections = document.section_titles()
        sections_list_md = "\n".join(map(lambda section: f' - {section}', sections))
        observations = observations + f'the retrieved page contains the following sections:\n{sections_list_md}'
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

def get_examples(chunk_size=EXAMPLES_CHUNK_SIZE):

    def mk_record(title, chunk_size):
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
            document_content, chunk_size=chunk_size)
        return ContentRecord(document, retrieval_history)

    colorado_orogeny_record = mk_record(
        'Colorado orogeny',
        chunk_size
    )

    high_plains_record = mk_record(
        'High Plains',
        chunk_size
    )

    high_plains_us_record = mk_record(
        'High Plains geology',
        chunk_size
    )

    milhouse_record = mk_record(
        'Milhouse Van Houten',
        chunk_size
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
        Question("What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?"),
        FunctionCall(
            "search",
            thought='I need to search Colorado orogeny, find the area that the eastern sector of the Colorado orogeny extends into, then find the elevation range of the area.',
            query="Colorado orogeny",
        ),
        FunctionResult('search', retrieval_observations(colorado_orogeny_record)),
        FunctionCall(
            'lookup',
            thought="It does not mention the eastern sector of the Colorado orogeny. I need to look up eastern sector.",
            keyword="eastern sector",
        ),
        FunctionResult('lookup', lookup_observations(colorado_orogeny_record.document, "eastern sector")),
        FunctionCall(
            'search',
            thought="The eastern sector of Colorado orogeny extends into the High Plains, so High Plains is the area. I need to search High Plains and find its elevation range.",
            query="High Plains",
        ),
        FunctionResult('search', retrieval_observations(high_plains_record)),
        FunctionCall(
            'search',
            thought='High Plains Drifter is a film. I need information about High Plains in geology or geography',
            query="High Plains geology",
        ),
        FunctionResult('search', retrieval_observations(high_plains_us_record)),
        *additional_messages,
        FunctionCall(
        'finish',
            thought='The High Plains have an elevation range from around 1,800 to 7,000 feet. I can use this information to answer the question about the elevation range of the area that the eastern sector of the Colorado orogeny extends into.',
            answer="approximately 1,800 to 7,000 feet",
        ),

        Question('Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?'),
        FunctionCall(
            'search',
            thought='I need to find out who Matt Groening named the Simpsons character Milhouse after.',
            query="Milhouse Simpson",
        ),
        FunctionResult( 'search', retrieval_observations(milhouse_record)),
        FunctionCall(
            'lookup',
            thought='The summary does not tell who Milhouse is named after, I should check the section called "Creation".',
            keyword="Creation",
        ),
        FunctionResult( 'lookup', lookup_observations(milhouse_record.document, "Creation")),
        FunctionCall(
            'finish',
            thought="Milhouse was named after U.S. president Richard Nixon, so the answer is President Richard Nixon.",
            answer="President Richard Nixon",
        ),
    ]


    return examples

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


if __name__ == "__main__":
    examples = get_examples()
    prompt = Prompt([
        system_message,
        *examples,
        Question("Bla bla bla"),
    ])

    from pprint import pprint
    pprint(prompt.openai_messages())
    print(prompt.plain())
    # print a line separator
    print()
    print("-" * 80)
    print()
    print("The lenght of the prompt is: " + str(len(prompt.plain())) + " characters.")
    print("The lenght of the prompt is: " + str(num_tokens_from_string(prompt.plain(), "cl100k_base")) + " tokens.")

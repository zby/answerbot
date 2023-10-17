system_prompt = '''
Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types:
(1) Search[entity], which searches the entity in wikipedia, it saves it for looking up parts of it and presents the beginning of it.
(2) Lookup[keyword], which returns the text surrounding the keyword in the current wikipedia page.
(3) Finish[answer], which returns the answer and finishes the task.
'''

prompt_messages = [
    { "role": "system",    "content": system_prompt },
    { "role": "user",      "content": "Question: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?" },
    { "role": "assistant", "content": """Thought: I need to search Colorado orogeny, find the area that the eastern sector of the Colorado orogeny extends into, then find the elevation range of the area.
Action: Search[Colorado orogeny]""" },
    { "role": "user",      "content": """Observation: Wikipedia search results for query: 'Colorado orogeny' is: 'Colorado orogeny', 'Laramide orogeny', 'Sevier orogeny', 'Colorado Mineral Belt', 'Variscan orogeny', 'Wyoming Craton', 'Timeline of natural history', 'Antler orogeny', 'Denver Basin', 'Sangre de Cristo Range'
Observation: Successfully retrieved 'Colorado orogeny' from Wikipedia.
Observation: the retrieved wikipedia page summary contains: The Colorado orogeny was an episode of mountain building (an orogeny) in Colorado and surrounding areas.This took place from 1780 to 1650 million years ago (Mya), during the Paleoproterozoic (Statherian Period). It is recorded in the Colorado orogen, a >500-km-wide belt of oceanic arc rock that extends southward into New Mexico. The Colorado orogeny was likely part of the larger Yavapai orogeny.""" },
    { "role": "assistant", "content": """
Thought: It does not mention the eastern sector of the Colorado orogeny. I need to look up eastern sector.
Action: Lookup[eastern sector]
""" },
    { "role": "user",      "content": """
Observation: == Description == The Colorado orogen, formerly called the Colorado province, is a >500-km-wide belt of oceanic arc rock (1.78–1.65 Ga) that extends southward into New Mexico and composes a major part of the Proterozoic provinces of southwestern United States. This transcontinental collisional event occurred during the Paleoproterozoic (Statherian Period). The Wyoming sector of the Colorado orogeny was formerly called the Medicine Bow orogeny. The eastern sector extends into the High Plains and is called the Central Plains orogeny. The boundary between the Colorado orogeny and the Wyoming craton is the Cheyenne belt, a 5-km-wide mylonitic shear zone that verges northward. The Cheyenne belt transects and cuts off the south edge of the older Trans-Hudson orogeny.""" },
    { "role": "assistant", "content": """
Thought: The eastern sector of Colorado orogeny extends into the High Plains, so High Plains is the area. I need to search High Plains and find its elevation range.
Action: Search[High Plains]
""" },
    { "role": "user", "content": """
Observation: Wikipedia search results for query: 'High Plains' is: 'High Plains', 'High Plains Drifter', 'High Plains (United States)', 'Great Plains', 'Ogallala Aquifer', 'High Plains Invaders', 'High Plains (Australia)', 'High Plains Reader', 'Llano Estacado', 'Bogong High Plains'
Observation: Title: 'High Plains' is ambiguous. Possible alternatives: High Plains (United States), High Plains (Australia), Altiplano (disambiguation), Western Plains (disambiguation)
Observation: Retrying with 'High Plains (United States)' (Retry 1/3).
Observation: Successfully retrieved 'High Plains (United States)' from Wikipedia.
Observation: the retrieved wikipedia page summary contains: The High Plains are a subregion of the Great Plains, mainly in the Western United States, but also partly in the Midwest states of Nebraska, Kansas, and South Dakota, generally encompassing the western part of the Great Plains before the region reaches the Rocky Mountains.The High Plains are located in eastern Montana, southeastern Wyoming, southwestern South Dakota, western Nebraska, eastern Colorado, western Kansas, eastern New Mexico, the Oklahoma Panhandle, and the Texas Panhandle. The southern region of the Western High Plains ecology region contains the geological formation known as Llano Estacado which can be seen from a short distance or on satellite maps. From east to west, the High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130 m).
""" },
    { "role": "assistant", "content": """
Thought: The elevation range of the High Plains is around 1,800 to 7,000 ft (550 to 2,130 m), so that is the answer.
Action: Finish[1,800 to 7,000 ft]
""" },
    { "role": "user", "content": """
Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
""" },
    { "role": "assistant", "content": """
Thought: The question simplifies to "The Simpsons" character Milhouse is named after who. I only need to search Milhouse and find who it is named after.
Action: Search[Milhouse]
""" },
    { "role": "user", "content": """
Observation: Wikipedia search results for query: 'Milhouse' is: 'Milhouse Van Houten', 'A Milhouse Divided', 'Pamela Hayden', 'Treehouse of Horror XIX', 'Bart's Friend Falls in Love', 'Milhouse of Sand and Fog', 'Milhouse Doesn't Live Here Anymore', 'Homer Defined', 'List of recurring The Simpsons characters', 'The Simpsons (season 35)'
Observation: Successfully retrieved 'Milhouse Van Houten' from Wikipedia.
Observation: the retrieved wikipedia page summary contains: Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening.Milhouse is Bart Simpson's best friend in Mrs. Krabappel's fourth grade class at Springfield Elementary School. He is insecure, gullible, and is often led into trouble by Bart, who takes advantage of his friend's naïveté. Milhouse is a regular target for school bully Nelson Muntz and his friends Jimbo Jones, Dolph Starbeam and Kearney Zzyzwicz. He has an unrequited crush on Bart's sister, Lisa, which is a common plot element. Milhouse debuted in the 1988 commercial "The Butterfinger Group" while The Simpsons was still airing as a cartoon short series on the Fox variety show The Tracey Ullman Show. When The Simpsons was greenlit for a full series by Fox, Milhouse became one of the series' most prominent recurring characters.
""" },
    { "role": "assistant", "content": """
Thought: The paragraph does not tell who Milhouse is named after, maybe I can look up "named after".
Action: Lookup[named after]
""" },
    { "role": "user", "content": """
Observation: Milhouse was named after U.S. president Richard Nixon, whose middle name was Milhous. 
""" },
    { "role": "assistant", "content": """
Thought: Milhouse was named after U.S. president Richard Nixon, so the answer is Richard Nixon.
Action: Finish[President Richard Nixon]
""" },
]

class TextualReactor:
    def __init__(self, query_function):
        self.messages = prompt_messages[:] # clone
        self.query_function = query_function

    def add_question(self, question):
        self.messages.append({ "role": "user", "content": f"Question: {question}" }) 

    def add_observation(self, observation):
        self.messages.append({ "role": "user", "content": f"Observation: {observation}" })

    def query(self):
        response = self.query_function(self.messages)
        self.messages.append(response) # XXX: validate the format of the content?
        print('response:', response)
        print("<<<", response["content"])
        response_lines = response["content"].strip().split('\n')
        last_line = response_lines[-1]

        if last_line.startswith('Action: Finish['):
            answer = last_line[15:-1]
            return ('finish', answer)
        elif last_line.startswith("Action: Search["):
            query = last_line[15:-1]
            return ('search', query)
        elif last_line.startswith("Action: Lookup["):
            query = last_line[15:-1]
            return ('lookup', query)


from prompt_builder import Prompt, PromptMessage, OpenAIMessage, User, System, Assistant, FunctionCall, FunctionResult

system_message = System('''
Solve a question answering task with interleaving Thought, Action, Observation steps. 
After each Observation you need to reflect on the response in a Thought step.
Thought can reason about the current situation, and Action means calling 
one of the available function.
'''

'''You are a helpful AI assistant trying to answer questions.
You analyze the question and available information and decide what to do next.
When you have enough information to answer the question please call the finish function with the answer.
When you need additional information please use the available functions to get it.
After each function call, please analyze the response reflect on it and decide what to do next.
''')

examples = [
    User("Question: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?"),
    FunctionCall(
        "search",
        thought='I need to search Colorado orogeny, find the area that the eastern sector of the Colorado orogeny extends into, then find the elevation range of the area.',
        query="Colorado orogeny",
    ),
    FunctionResult(
        'search',
        "Wikipedia search results for query: 'Colorado orogeny' is: 'Colorado orogeny', 'Laramide orogeny', "
        "'Colorado Mineral Belt', 'Sevier orogeny', 'Geology of Colorado', 'Wyoming Craton', 'Timeline of "
        "natural history', 'Variscan orogeny', 'Sangre de Cristo Range', 'Colorado Plateau'\n"
        "Successfully retrieved 'Colorado orogeny' from Wikipedia.\n"
        'The retrieved wikipedia page summary starts with: The Colorado orogeny was an episode of mountain '
        'building (an orogeny) in Colorado and surrounding areas.This took place from 1780 to 1650 million '
        'years ago (Mya), during the Paleoproterozoic (Statherian Period). It is recorded in the Colorado '
        'orogen, a >500-km-wide belt of oceanic arc rock that extends southward into New Mexico. The Colorado '
        'orogeny was likely part of the larger Yavapai orogeny.',
    ),
    FunctionCall(
        'lookup',
        thought="It does not mention the eastern sector of the Colorado orogeny. I need to look up eastern sector.",
        keyword="eastern sector",
    ),
    FunctionResult(
        'lookup',
        "== Description == The Colorado orogen, formerly called the Colorado province, is a >500-km-wide belt of oceanic arc rock (1.78–1.65 Ga) that extends southward into New Mexico and composes a major part of the Proterozoic provinces of southwestern United States. This transcontinental collisional event occurred during the Paleoproterozoic (Statherian Period). The Wyoming sector of the Colorado orogeny was formerly called the Medicine Bow orogeny. The eastern sector extends into the High Plains and is called the Central Plains orogeny. The boundary between the Colorado orogeny and the Wyoming craton is the Cheyenne belt, a 5-km-wide mylonitic shear zone that verges northward. The Cheyenne belt transects and cuts off the south edge of the older Trans-Hudson orogeny."
    ),
    FunctionCall(
        'search',
        thought="The eastern sector of Colorado orogeny extends into the High Plains, so High Plains is the area. I need to search High Plains and find its elevation range.",
        query="High Plains",
    ),
    FunctionResult(
        'search',
        "Wikipedia search results for query: 'High Plains' is: 'High Plains Drifter', 'High Plains', 'High "
        "Plains (United States)', 'Ogallala Aquifer', 'Great Plains', 'High Plains (Australia)', 'High Plains "
        "Reader', 'High Plains Invaders', 'Bogong High Plains', 'Llano Estacado'\n"
        "Successfully retrieved 'High Plains Drifter' from Wikipedia.\n"
        'The retrieved wikipedia page summary starts with: High Plains Drifter is a 1973 American Western film '
        'directed by Clint Eastwood, written by Ernest Tidyman, and produced by Robert Daley for The Malpaso '
        'Company and Universal Pictures.The film stars Eastwood as a mysterious stranger who metes out justice '
        "in a corrupt frontier mining town. The film was influenced by the work of Eastwood's two major "
        'collaborators, film directors Sergio Leone and Don Siegel. In addition to Eastwood, the film also '
        'co-stars Verna Bloom, Mariana Hill, Mitchell Ryan, Jack Ging, and Stefan Gierasch. The film was shot '
        'on location on the shores of Mono Lake, California. Dee Barton wrote the film score. The film was '
        'critically acclaimed at the time of its initial release and remains popular.',
    ),
    FunctionCall(
        'search',
        thought='High Plains Drifter is a film. I need information about different High Plains',
        query="High Plains elevation range",
    ),
    FunctionResult(
        'search',
        "Wikipedia search results for query: 'High Plains elevation range' is: 'High Plains (United States)', "
        "'Laramie Plains', 'Plain', 'Roaring Plains West Wilderness', 'Northern Basin and Range ecoregion', "
        "'Great Basin Desert', 'List of elevation extremes by country', 'Liverpool Plains', 'Plateau', "
        "'Geography of the United States'\n"
        "Successfully retrieved 'High Plains (United States)' from Wikipedia.\n"
        'The retrieved wikipedia page summary starts with: The High Plains are a subregion of the Great Plains, '
        'mainly in the Western United States, but also partly in the Midwest states of Nebraska, Kansas, and '
        'South Dakota, generally encompassing the western part of the Great Plains before the region reaches '
        'the Rocky Mountains.The High Plains are located in eastern Montana, southeastern Wyoming, southwestern '
        'South Dakota, western Nebraska, eastern Colorado, western Kansas, eastern New Mexico, the Oklahoma '
        'Panhandle, and the Texas Panhandle. The southern region of the Western High Plains ecology region '
        'contains the geological formation known as Llano Estacado which can be seen from a short distance or '
        'on satellite maps. From east to west, the High Plains rise in elevation from around 1,800 to 7,000 ft '
        '(550 to 2,130 m).',
    ),
    FunctionCall(
        'finish',
        thought='The High Plains have an elevation range from around 1,800 to 7,000 feet. I can use this '
        'information to answer the question about the elevation range of the area that the eastern sector of '
        'the Colorado orogeny extends into.',
        answer="The elevation range for the area that the eastern sector of the Colorado orogeny extends into is approximately 1,800 to 7,000 feet.",
    ),

    User('Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?'),
    FunctionCall(
        'search',
        thought='I need to find out who Matt Groening named the Simpsons character Milhouse after.',
        query="Milhouse Simpson",
    ),
    FunctionResult(
        'search',
        "Wikipedia search results for query: 'Milhouse Simpson' is: 'Milhouse Van Houten', 'A Milhouse "
        "Divided', 'Bart Simpson', 'The Simpsons', 'List of recurring The Simpsons characters', 'Radioactive "
        "Man (The Simpsons episode)', 'Milhouse of Sand and Fog', 'Treehouse of Horror XIX', 'The Simpsons "
        "(season 35)', 'Homer Simpson'\n"
        "Successfully retrieved 'Milhouse Van Houten' from Wikipedia.\n"
        'The retrieved wikipedia page summary starts with: Milhouse Mussolini Van Houten is a recurring character '
        'in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt '
        "Groening.Milhouse is Bart Simpson's best friend in Mrs. Krabappel's fourth grade class at Springfield "
        'Elementary School. He is insecure, gullible, and is often led into trouble by Bart, who takes '
        "advantage of his friend's naïveté. Milhouse is a regular target for school bully Nelson Muntz and his "
        "friends Jimbo Jones, Dolph Starbeam and Kearney Zzyzwicz. He has an unrequited crush on Bart's sister, "
        'Lisa, which is a common plot element. Milhouse debuted in the 1988 commercial "The Butterfinger Group" '
        'while The Simpsons was still airing as a cartoon short series on the Fox variety show The Tracey '
        'Ullman Show. When The Simpsons was greenlit for a full series by Fox, Milhouse became one of the '
        "series' most prominent recurring characters.",
    ),
    FunctionCall(
        'lookup',
        thought='The summary does not tell who Milhouse is named after, I should check the section called "Creation".',
        keyword="Creation",
    ),
    FunctionResult(
        'lookup',
        "Milhouse was named after U.S. president Richard Nixon, whose middle name was Milhouse.",
    ),
    FunctionCall(
        'finish',
        thought="Milhouse was named after U.S. president Richard Nixon, so the answer is President Richard Nixon.",
        answer="President Richard Nixon",
    ),
]

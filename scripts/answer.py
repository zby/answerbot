import logging
import litellm

from dotenv import load_dotenv

from answerbot.tools.wiki_tool import WikipediaTool
from answerbot.qa_processor import QAProcessor, QAProcessorNew

# Set logging level to INFO for qa_processor.py
#logging.getLogger('qa_processor').setLevel(logging.INFO)
qa_logger = logging.getLogger('qa_processor')
qa_logger.setLevel(logging.INFO)

# Create a console handler
#console_handler = logging.StreamHandler()

# Add the handler to the logger
#qa_logger.addHandler(console_handler)



load_dotenv()
litellm.success_callback = ["langfuse"]
litellm.failure_callback = ["langfuse"]
#litellm.success_callback=["helicone"]
#litellm.set_verbose=True

#client = OpenAI(
#    api_key=config['OPENAI_API_KEY'],
#    timeout=httpx.Timeout(70.0, read=60.0, write=20.0, connect=6.0)
#)

if __name__ == "__main__":

    # question = "What was the first major battle in the Ukrainian War?"
    # question = "What were the main publications by the Nobel Prize winner in economics in 2023?"
    # question = "What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?"
    # question = 'Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?'
    # question = "how old was Donald Tusk when he died?"
    # question = "how many keys does a US-ANSI keyboard have on it?"
    # question = "How many children does Donald Tusk have?"
    # question = "The director of the romantic comedy \"Big Stone Gap\" is based in what New York city?"
    #question = "When Poland became elective monarchy?"
    #question = "Were Scott Derrickson and Ed Wood of the same nationality?"
    #question = "What science fantasy young adult series, told in first person, has a set of companion books narrating the stories of enslaved worlds and alien species?"
    # question = "What year did Guns N Roses perform a promo for a movie starring Arnold Schwarzenegger as a former New York Police detective?"
    #question = "What is the weight proportion of oxygen in water?"
    #question = "Czy dane kardy kredytowej są danymi osobowymi w Polsce"
    #question = "How much is two plus two"
    #question = "Who is older, Annie Morton or Terry Richardson?"

    #question = "What are the concrete steps proposed to ensure AI safety?"
    #question = 'What are the steps required to authorize the training of generative AI?'

    #question = "What is the name of the fight song of the university whose main campus is in Lawrence, Kansas and whose branch campuses are in the Kansas City metropolitan area?"
    question = "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?"
    #question = "The arena where the Lewiston Maineiacs played their home games can seat how many people?"
    #question = "What is the seating capacity of Androscoggin Bank Colisée?"
    #question = "Who portrayed Corliss Archer in the film Kiss and Tell?"


    app = QAProcessorNew(
        toolbox=[WikipediaTool(chunk_size=400)],
        max_iterations=5,
        model='gpt-3.5-turbo',
        #model='gpt-4o',
        #model='claude-3-5-sonnet-20240620',
        #model="claude-3-haiku-20240307",
        prompt_templates_dirs=['answerbot/templates/common', 'answerbot/templates/wiki_researcher'],
        fail_on_tool_error=True
    )

    print()
    print(app.process(question))

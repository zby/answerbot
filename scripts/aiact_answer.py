from answerbot.llm import LLM
from answerbot.aiact import AiActReactor, format_results



if __name__ == '__main__':
    question = '''
    How is transparency defined in the AI Act and what transparency requirements apply to low-risk Ai systems?
    '''

    llm = LLM(model='gpt-3.5-turbo-0125')
    llm = LLM(model='gpt-4-turbo')
    reactor = AiActReactor(llm=llm, question=question,energy=200)
    result = reactor()

    print(format_results(result))

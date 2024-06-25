import logging
import os
import httpx
from openai import OpenAI
from answerbot.react import LLMReactor 
from answerbot.tools.paged_owu import Catalog
from pprint import pprint
import click
import json
import litellm

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)

#litellm.success_callback=["helicone"]
#litellm.set_verbose=True



question = '''
Jaka jest suma ubezpieczenie w ryzyku szyb samochodowych?
'''



@click.command()
@click.option('--local', '-l', type=str, default=None)
@click.option('--max-llm-calls', '-m', type=int, default=7)
def main(local: str|None=None, max_llm_calls: int=11):
    
    catalog = Catalog("data/paged_OWU/parsed_files")
    
    def _sys_prompt(max_llm_calls):
        return f"""
        Please answer the following question, using the catalog tool available.
        You only have {max_llm_calls-1} to the tools.
        Some documents have a table of content near the start of the document - you can use it to navigate
        the document. But please take into account that the page numbers in the table of content are not
        the same as the page numbers in the document. There are pages at the start of the document that are
        not included in the table of content - so the page numbers in the table of content are shifted by a couple
        of pages.
        
        At the start please always retrieve the first 4 pages - because there are important informations at them.
        
        There are following documents in the catalog:
        
        {catalog.format_catalog()}
        """


    reactor = LLMReactor(
            model='claude-3-5-sonnet-20240620',
            toolbox=[catalog.get_document_by_filename, catalog.get_page],
            max_llm_calls=max_llm_calls,
            get_system_prompt=_sys_prompt,
            )

    trace = reactor.process(question)
    print(trace.generate_report())
    print()
    print(str(trace.what_have_we_learned))
    print()
    pprint(trace.soft_errors)


if __name__ == '__main__':
    main()

import json
import logging
import os

from answerbot.tools.euact import frontmatter

logging.basicConfig(level=logging.INFO)

import click
from dotenv import load_dotenv

from answerbot.react import LLMReactor
from answerbot.tools.euact.document import DocumentParseError, EuActWebOpener, IdOpener, FMDOpener, MetaOpener
from answerbot.tools.euact.toc import TocEntry, fetch_toc, format_toc_entry, iter_toc
from answerbot.tools.euact.tool import TocTool, replace_markdown_urls
from pprint import pprint
import re

load_dotenv()


SANITIZER = re.compile(r'[^a-zA-Z0-9]')


question = '''
How is transparency defined in the AI Act and what transparency requirements apply to low-risk Ai systems?
'''


def _sys_prompt(max_llm_calls):
    return f"""
    Please answer the following question, using the tool available.
    You only have {max_llm_calls-1} to the tools.
    """


@click.group
def main():
    pass


@click.command
@click.argument('target', type=str)
def download(target: str):
    os.makedirs(target, exist_ok=True)
    opener = EuActWebOpener()

    toc = fetch_toc()

    for entry in iter_toc(toc):
        url = entry['url']
        title = entry['title']
        if entry['children']:
            continue
        if not isinstance(url, str):
            continue
        if not isinstance(title, str):
            continue
        try:
            document = opener(url) 
        except DocumentParseError:
            continue
        fm = frontmatter.dump(document.text, document.meta)
        title_sanitized = sanitize_string(title)
        filename = f'{title_sanitized}.md'
        filepath = os.path.join(target, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(fm)

    toc_filepath = os.path.join(target, 'eu_ai_act_toc.json')
    with open(toc_filepath, 'w', encoding='utf-8') as f:
        json.dump(toc, f)



def sanitize_string(src: str) -> str:
    return SANITIZER.sub('-', src)


@click.command()
@click.option('--max-llm-calls', '-m', type=int, default=7)
@click.option('--local', '-l', type=str)
def answer(
        max_llm_calls: int=7,
        local: str|None=None,
        ):

    if local and os.path.exists(os.path.join(local, 'eu_ai_act_toc.json')):
        with open(os.path.join(local, 'eu_ai_act_toc.json'), 'r', encoding='utf-8') as f:
            toc = json.load(f)
    else:
        toc = fetch_toc()

    fmd_opener = FMDOpener({'t': local} if local else {})
    meta_opener = MetaOpener([EuActWebOpener(), fmd_opener])
    id_opener = IdOpener(meta_opener)

    urls = [entry['url'] for entry in iter_toc(toc) if entry['url'] and not entry['children']]
    urlmap = {}

    fmd_opener = FMDOpener()

    if local:
        for filename in os.listdir(local):
            if not filename.endswith('.md'):
                continue
            path = os.path.join(local, filename)
            with open(path, 'r', encoding='utf-8') as f:
                _, meta = frontmatter.load(f.read())
            url = meta.get('url')
            if not url:
                continue
            urlmap[url] = f'fmd://t/{filename}'

    id_opener.fill(urls)  # type: ignore
    id_opener.fill(urlmap.values())  
    urlmap = {**id_opener.urlmap(), **urlmap}
    
    reactor = LLMReactor(
            model='gpt-4o',
            toolbox=[
                TocTool(toc, id_opener, urlmap)
                ],
            max_llm_calls=max_llm_calls,
            get_system_prompt=_sys_prompt,
            question_checks=[],
            )

    trace = reactor.process(question)
    print(trace.generate_report())
    print()
    print(str(trace.what_have_we_learned))
    print()
    pprint(trace.soft_errors)


main.add_command(answer)
main.add_command(download)

if __name__ == '__main__':
    main()

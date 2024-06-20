from answerbot.tools.aaext import get_eu_act_toc_raw, retrieve, sanitize_string
import os
import json
import click


@click.command()
@click.argument('folder')
def main(folder: str):
    os.makedirs(folder, exist_ok=True)
    raw = get_eu_act_toc_raw()
    toc_filename = os.path.join(folder, 'toc.json')

    with open(toc_filename, 'w', encoding='utf-8') as f:
        json.dump(raw, f)

    download_from_section(folder, raw)


def download_from_section(folder: str, items: list[dict]):
    for item in items:
        children: list|None = item.get('children')
        url = item['url']
        title: str|None = item.get('title')

        if children:
            download_from_section(folder, children)
        
        if url and title and not children:
            sanitized_title = sanitize_string(title)
            filepath = os.path.join(folder, f'{sanitized_title}.md')
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(retrieve(url))


if __name__ == '__main__':
    main()

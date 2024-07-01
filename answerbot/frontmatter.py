"""

Tools to parse and dump markdown documents with metadata written as
[front matter](https://jekyllrb.com/docs/front-matter/)

"""


from typing import Any
import yaml
import re


_FRONTMATTER_DOCUMENT_RE = re.compile(r'^---\s*\n(.*?)\n---\s*\n(.*)', re.DOTALL)


def fm_dumps(content: str, meta: dict[str, Any]) -> str:
    if not meta:
        return content
    metadata_str = yaml.dump(meta, default_flow_style=False)
    return f"---\n{metadata_str}---\n{content}"


def fm_parse(src: str) -> tuple[str, dict[str, Any]]:
    match = _FRONTMATTER_DOCUMENT_RE.match(src)

    if not match:
        return src, {}

    metadata_str, content = match.groups()
    metadata = yaml.safe_load(metadata_str)
    return content, metadata



if __name__ == '__main__':
    document = '''---
title: "How to Add Metadata to a Markdown Document"
author: "Jane Doe"
date: "2024-07-01"
tags: ["markdown", "tutorial", "metadata"]
---
# How to Add Metadata to a Markdown Document


Metadata helps organize and manage Markdown documents more effectively...
    '''

    content, meta = fm_parse(document) 

    print(f'*** parsed ***:\nmetadata:{meta}\ncontent:\n{content}')
    print(f'*** parsed w/o metadata ***\n{fm_parse(content)}')
    dumped = fm_dumps(content, meta)
    print(f'*** dumped with metadata ***\n{dumped}')
    print(f'*** dumped w/o metadata ***\n{fm_dumps(content, {})}')
    content_meta = fm_parse(dumped)
    print(f'*** parsed back ***:\nmetadata:{meta}\ncontent:\n{content}')

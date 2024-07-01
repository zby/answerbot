from typing import Annotated, Callable

from llm_easy_tools import get_tool_defs
from answerbot.tools.observation import InfoPiece, Observation
from .toc import TocEntry, format_toc_entry
from .document import DocumentParseError, Opener
from . import frontmatter
from logging import getLogger
import re


MARKDOWN_LINK_RE = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')


class TocTool:
    def __init__(
            self,
            toc: TocEntry,
            opener: Opener,
            url_map: dict[str, str],
            ) -> None:
        self._toc = toc
        self._opener = opener
        self._url_map = url_map
        self._seen = set()

    def get_llm_tools(self) -> list[Callable]:
        def open_url(url: Annotated[str, 'url']):
            return self.open_url(url)

        schemas = '\n'.join(f'- {schema}' for schema in self._opener.schemas())
        open_url.__doc__ = (
                'Open a document at `url` as markdown\n\n'
                f'Available url schemas are: {schemas}\n\n'
                'Do not try to open an url that does not match any of the schemas '
                'and if possible avoid opening urls not found in table of contents, opened documents, or metadata'
                )

        return [
                self.show_table_of_contents,
                open_url,
                ]

    def show_table_of_contents(self) -> Observation:
        """ Show EU artificial intelligence act's table of contents in markdown format """

        toc_raw = format_toc_entry(self._toc)
        toc_links_replaced = self._replace_markdown_urls(toc_raw)
        print(toc_links_replaced)
        return Observation([
            InfoPiece('Opening table of contents'),
            InfoPiece(toc_links_replaced)
            ])

    def open_url(self, url: Annotated[str, 'url']) -> Observation:
        """ Open a document at `url` as markdown """

        getLogger(__name__).info(f'Opening url: {url}')

        if url in self._seen:
            getLogger(__name__).error(f'{url} already seen')
            return Observation([InfoPiece('You have already seen this article')])

        if not self._opener.url_valid(url):
            getLogger(__name__).error(f'{url} is not a valid url')
            return Observation([InfoPiece(f'Not a valid url (does not match any of the schemas): {url}')])

        try:
            doc = self._opener(url)
        except DocumentParseError:
            return Observation([InfoPiece('Document empty')])
        content, meta = doc.text, doc.meta
        doc_url = meta.get('url')

        meta_formatted_raw = frontmatter.dump_fm(meta)
        meta_formatted_urls_replaced = self._replace_markdown_urls(meta_formatted_raw)
        content_urls_replaced = self._replace_markdown_urls(content)

        result = f'---\n{meta_formatted_urls_replaced}\n---\n{content_urls_replaced}'

        getLogger(__name__).info(f'Document retrieved {url}: {result}')

        return Observation([
            InfoPiece(result, quotable=True, source=doc_url)
            ], current_url=doc_url, available_tools=self._available_tools())  # type: ignore

    def _available_tools(self) -> list[str]:
        tools = self.get_llm_tools()
        schemas = get_tool_defs(tools)
        result = [
                f"{schema['function']['name']}: {schema['function']['description']}"   # type: ignore
                for schema in schemas
                ]
        return result

    def _replace_markdown_urls(self, src: str) -> str:
        url_map = {}

        # dirty fix for trailing /
        for k, v in self._url_map.items():
            if k.endswith('/'):
                k_ = k.rstrip('/')
                if k_ not in self._url_map:
                    url_map[k_] = v
            else:
                k_ = k + '/'
                if k_ not in self._url_map:
                    url_map[k_] = v
            url_map[k] = v

        return replace_markdown_urls(src, lambda k: url_map.get(k, k))


def replace_markdown_urls(src: str, func: Callable[[str], str]):
    result = MARKDOWN_LINK_RE.sub(
            lambda match: f'[{match.group(1)}]({func(match.group(2))})', 
            src
            )
    if result != src:
        return replace_markdown_urls(result, func)
    return result




from dataclasses import dataclass, field
import os
from typing import Any, Collection, Sequence, Iterator

from bs4 import BeautifulSoup, Tag
from markdownify import markdownify
import requests
from urllib.parse import urlparse

from . import frontmatter
import os
from random import Random
from string import ascii_lowercase
from abc import ABC, abstractmethod
from logging import getLogger


class DocumentParseError(Exception):
    pass


def _short_str_ids(k=6, seed=0) -> Iterator[str]:
    random = Random(seed)
    while True:
        yield ''.join(random.choices(ascii_lowercase, k=k))


@dataclass(frozen=True)
class RetrievedDocument:
    text: str
    meta: dict[str, Any]


class Opener(ABC):
    @abstractmethod
    def schemas(self) -> list[str]:
        ...


    @abstractmethod
    def url_valid(self, url: str) -> bool:
        ...

    @abstractmethod
    def __call__(self, url: str) -> RetrievedDocument:
        ...


@dataclass
class MetaOpener(Opener):
    openers: Sequence[Opener]

    def schemas(self) -> list[str]:
        result = []
        for opener in self.openers:
            result.extend(opener.schemas())
        return result

    def url_valid(self, url: str) -> bool:
        return any(x.url_valid(url) for x in self.openers)

    def __call__(self, url: str) -> RetrievedDocument:
        getLogger().info(f'MetaOpener: {url}')
        return next(x(url) for x in self.openers if x.url_valid(url))



@dataclass
class IdOpener(Opener):
    opener: Opener
    generator: Iterator[str] = field(default_factory=_short_str_ids)
    ids: dict[str, str] = field(default_factory=dict)

    def schemas(self) -> list[str]:
        return ['id://...', *self.opener.schemas()]

    def __generate(self) -> str:
        return next(id_ for id_ in self.generator if id_ not in self.ids)

    def fill(self, values: Collection[str]):
        for value in values:
            self.add(value)

    def add(self, value: str) -> str:
        id_ = self.__generate()
        self.ids[id_] = value
        return id_

    def replace(self, key: str, value: str):
        if key not in self.ids:
            raise KeyError(key)
        self.ids[key] = value

    def urlmap(self) -> dict[str, str]:
        return {v: f'id://{k}' for k, v in self.ids.items()}

    def url_valid(self, url: str) -> bool:
        if self.opener.url_valid(url):
            return True

        p = urlparse(url)

        if p.scheme != 'id':
            return False

        if not p.netloc:
            return False

        if p.params or p.query or p.fragment:
            return False

        return p.netloc in self.ids


    def __call__(self, url: str) -> RetrievedDocument:
        getLogger().info(f'IdOpener: {url}')

        if self.opener.url_valid(url):
            return self.opener(url)

        p = urlparse(url)

        return self.opener(self.ids[p.netloc])


@dataclass
class FMDOpener(Opener):
    netlocs: dict[str, str] = field(default_factory=dict)

    def schemas(self) -> list[str]:
        return ['fmd://...']

    def url_valid(self, url: str) -> bool:
        return urlparse(url).scheme == 'fmd'

    def __call__(self, url: str) -> RetrievedDocument:
        getLogger().info(f'FMDOpener: {url}')
        parsed_url = urlparse(url)
        
        netloc = self.netlocs.get(parsed_url.netloc, parsed_url.netloc)

        if netloc:
            path = os.path.abspath(os.path.join(netloc, parsed_url.path.lstrip('/')))
        else:
            path = os.path.abspath(parsed_url.path)

        with open(path, 'r', encoding='utf-8') as f:
            document = f.read()

        content, meta = frontmatter.load(document)

        return RetrievedDocument(content, meta)


class EuActWebOpener(Opener):
    def schemas(self) -> list[str]:
        return ['https://artificialintelligenceact.eu/...']

    def url_valid(self, url: str) -> bool:
        p = urlparse(url)

        if p.scheme not in ('http', 'https'):
            return False

        if p.netloc != 'artificialintelligenceact.eu':
            return False

        return True

    def __call__(self, url: str) -> RetrievedDocument:
        getLogger().info(f'EuActWebOpener: {url}')
        meta = {'url': url}

        response = requests.get(url)
        response.raise_for_status()
        html = response.text

        soup = BeautifulSoup(html, 'html.parser')

        if soup.title and soup.title.string:
            meta = {**meta, 'title': soup.title.string.replace('\n', ' ')}
        
        paragraphs = [
                markdownify(str(p))
                for p in soup.select('div.et_pb_post_content > p')
                ]


        if not paragraphs:
            raise DocumentParseError('Document empty')

        recitals = []

        for recital_link in soup.select('div.recitals-list > a'):
            if not isinstance(recital_url := recital_link['href'], str):
                continue
            if not isinstance(
                    recital_p := recital_link.select_one('div.recital-wrapper > p.related-recital'), 
                    Tag):
                continue
            recitals.append((recital_url.strip(), recital_p.text.replace('\n', '').strip()))

        content = ''.join(paragraphs)
        recitals = [f'[{title}]({url})' for url, title in recitals]

        meta = {**meta, 'Suitable recitals': recitals} if recitals else meta

        return RetrievedDocument(content, meta)




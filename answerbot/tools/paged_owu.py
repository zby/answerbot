import json
import os
from dataclasses import dataclass
from typing import List, Dict, Optional, Union

from answerbot.tools.observation import Observation

@dataclass(frozen=True)
class Document:
    metadata: Dict
    content: List[Dict]
    title: str

    def get_page(self, page_number: int) -> Union[Observation, str]:
        if 1 <= page_number <= len(self.content):
            page_content = self.content[page_number - 1]['md']
            return Observation(
                content=page_content,
                source=self.metadata['filename'],
                operation="get_page",
                quotable=True
            )
        else:
            error_message = f"Error: Page {page_number} does not exist. Valid page numbers are 1 to {len(self.content)}."
            return error_message

class Catalog:
    def __init__(self, directory: str):
        self.directory = directory
        self.documents_by_filename: Dict[str, Document] = {}
        self.documents_by_title: Dict[str, Document] = {}
        self.current_document: Optional[Document] = None
        self._load_documents()

    def _load_documents(self):
        for filename in os.listdir(self.directory):
            if filename.endswith('.json'):
                full_path = os.path.join(self.directory, filename)
                doc = self._load_document(full_path)
                self.documents_by_filename[filename] = doc
                self.documents_by_title[doc.title] = doc

    @staticmethod
    def _load_document(filename: str) -> Document:
        with open(filename, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        title = Catalog._find_title(data['content'][0]['md'])
        return Document(metadata=data['metadata'], content=data['content'], title=title)

    @staticmethod
    def _find_title(first_page: str) -> str:
        lines = first_page.split('\n')
        title_lines = []
        empty_lines_count = 0
        header_found = False

        for line in lines:
            line = line.strip()
            if line.startswith('# '):
                title_lines.append(line.lstrip('# ').strip())
                empty_lines_count = 0
                header_found = True
            elif header_found:
                if not line:
                    empty_lines_count += 1
                    if empty_lines_count > 1:
                        break
                elif line.startswith('# '):
                    title_lines.append(line.lstrip('# ').strip())
                    empty_lines_count = 0
                else:
                    break

        if title_lines:
            return ' '.join(title_lines)

        # Apply other heuristics here if needed
        
        return "Untitled Document"

    def get_document_by_filename(self, name: str) -> Optional[Document]:
        """
        Loads a document by its filename.
        """
        print(f"Getting document by filename: {name}")
        self.current_document = self.documents_by_filename.get(name)
        return """Document retrieved and saved. You can now use the get_page tool to get the content of the pages.
"""

    def get_document_by_title(self, title: str) -> Optional[Document]:
        self.current_document = self.documents_by_title.get(title)
        return "Document retrieved and saved. You can now use the get_page tool to get the content of the pages."

    def get_page(self, page_number: int) -> Union[Observation, str]:
        """
        Returns the content of the page with the given number.
        """
        print(f"Getting page: {page_number}")
        if self.current_document is None:
            return "No document is currently loaded."
        
        return self.current_document.get_page(page_number)

    def format_catalog(self) -> str:
        """
        Returns a formatted string representation of the catalog.
        """
        catalog_str = "Available documents:\n"
        for idx, (filename, doc) in enumerate(self.documents_by_filename.items(), 1):
            catalog_str += f"{idx}. {filename}\n"
            catalog_str += f"   Title: {doc.title}\n"
            catalog_str += f"   SHA256: {doc.metadata['sha256']}\n\n"
        return catalog_str.strip()

# Usage example:
if __name__ == "__main__":
    catalog = Catalog("data/paged_OWU/parsed_files")
    print(catalog.format_catalog())

    # Get a specific document by title
    doc_filename = 'Ogolne_Warunki_Ubezpieczenia_Generali_z_mysla_o_podrozy_obowiazujace_od_25_05_2023_r_b4f387665f.json'
    doc = catalog.get_document_by_filename(doc_filename)
    if catalog.current_document:
        print(f"Document filename: {catalog.current_document.metadata['filename']}")
        print(f"Document title: {catalog.current_document.title}")
        print(f"Page 1 content:\n{catalog.get_page(1)}")
    else:
        print("Document not found.")
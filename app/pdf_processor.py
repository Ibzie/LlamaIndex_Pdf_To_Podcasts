# from llama_index.core import Document
# from llama_index.core.node_parser import SimpleNodeParser
# from llama_index.readers.file import PDFReader
# from typing import List

from llama_index.core import Document
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.readers.file import PDFReader
from typing import List

class PDFProcessor:
    def __init__(self):
        self.parser = SimpleNodeParser.from_defaults(
            chunk_size=500,
            chunk_overlap=50
        )
        self.reader = PDFReader()

    def process_pdf(self, pdf_path: str) -> List[Document]:
        try:
            documents = self.reader.load_data(pdf_path)
            nodes = self.parser.get_nodes_from_documents(documents)
            return nodes
        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")
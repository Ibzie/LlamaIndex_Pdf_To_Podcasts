from llama_index.core import Document
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.readers.file import PDFReader
from typing import List
import os
from pathlib import Path
import concurrent.futures

class PDFProcessor:
   def __init__(self):
       self.parser = SimpleNodeParser.from_defaults(
           chunk_size=500,
           chunk_overlap=50
       )
       self.reader = PDFReader()

   def process_pdf(self, pdf_path: str) -> List[Document]:
       abs_path = str(Path(pdf_path).resolve())
       if not os.path.exists(abs_path):
           raise FileNotFoundError(f"PDF file not found at: {abs_path}")
           
       try:
           with concurrent.futures.ThreadPoolExecutor() as executor:
               future = executor.submit(self.reader.load_data, abs_path)
               documents = future.result()
               
           nodes = self.parser.get_nodes_from_documents(documents)
           return nodes
       except Exception as e:
           raise Exception(f"Error processing PDF: {str(e)}")
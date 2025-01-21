import os
from dotenv import load_dotenv
from pathlib import Path

class Config:
    def __init__(self):
        # Load .env from root directory
        root_dir = Path(__file__).parent.parent
        load_dotenv(root_dir / '.env')
        
        # API Keys
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        self.pdf_path = os.getenv("PDF_PATH", os.path.join(root_dir, "Data/input.pdf"))
        self.refrence_audio_path = os.path.join(root_dir, "Data/reference_voices/")
        self.text_output_path = os.getenv("TEXT_OUTPUT_PATH", os.path.join(root_dir, "Data/output.txt"))
        print(f"Current file location: {Path(__file__)}")
        print(f"Root directory: {root_dir}")
        print(f"Env file path: {root_dir / '.env'}")
        print(f"PDF path: {os.path.join(root_dir, 'Data/input.pdf')}")


Config()

        


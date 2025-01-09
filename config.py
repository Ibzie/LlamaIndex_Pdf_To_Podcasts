# config.py
import os
from dotenv import load_dotenv

class Config:
    def __init__(self):
        load_dotenv()
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        self.hugging_face_token = os.getenv("HUGGING_FACE_TOKEN")
        if not self.hugging_face_token:
            raise ValueError("HUGGING_FACE_TOKEN not found in environment variables")
        
        self.pdf_path = os.getenv("PDF_PATH", "input.pdf")
        self.text_output_path = os.getenv("TEXT_OUTPUT_PATH", "output.txt")
        self.audio_output_path = os.getenv("AUDIO_OUTPUT_PATH", "podcast.mp3")
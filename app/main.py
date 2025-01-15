from pathlib import Path
from .pdf_processor import PDFProcessor
from .conversation_generator import ConversationGenerator
from .audio_generator import AudioGenerator
from config import Config
import os
import torch

def main():
    try:
        config = Config()
        
        use_gpu = torch.cuda.is_available()
        print("Initializing audio generator...")
        audio_generator = AudioGenerator(use_gpu=use_gpu)

        print("\nUsing", "GPU" if use_gpu else "CPU", "for audio generation")

        if os.path.exists(config.text_output_path):
            print(f"\nFound existing text conversation at {config.text_output_path}")
            print("Skipping text generation...")
            
            with open(config.text_output_path, 'r', encoding='utf-8') as f:
                full_text = f.read()
        else:
            pdf_processor = PDFProcessor()
            conversation_generator = ConversationGenerator(config.groq_api_key)
            
            print("\nProcessing PDF...")
            nodes = pdf_processor.process_pdf(config.pdf_path)
            print(f"Found {len(nodes)} sections in PDF")

            print("\nGenerating conversations...")
            conversations = []
            for i, node in enumerate(nodes, 1):
                print(f"Processing section {i}/{len(nodes)}")
                conversation = conversation_generator.generate_conversation(
                    chunk=node.text,
                    is_first_segment=(i == 1)
                )
                conversations.append(conversation)

            conversation_generator.save_conversations(conversations, config.text_output_path)
            print(f"\nText conversation saved to {config.text_output_path}")
            
            full_text = "".join(conversations)
        
        print("\nGenerating audio podcast...")
        audio_generator.generate_podcast(
            text=full_text,
            output_path="",  # Removed as it's handled by AudioGenerator
            batch_size=3
        )

    except Exception as e:
        print(f"\nError: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
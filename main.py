from pdf_processor import PDFProcessor
from conversation_generator import ConversationGenerator
from audio_generator import AudioGenerator
from config import Config
import atexit
import os

def main():
    try:
        # Initialize components
        config = Config()
        audio_generator = AudioGenerator()

        # Check if output.txt already exists
        if os.path.exists(config.text_output_path):
            print(f"\nFound existing text conversation at {config.text_output_path}")
            print("Skipping text generation...")
            
            # Read existing conversation
            with open(config.text_output_path, 'r', encoding='utf-8') as f:
                full_text = f.read()
        else:
            # Initialize components needed for text generation
            pdf_processor = PDFProcessor()
            conversation_generator = ConversationGenerator(config.groq_api_key)
            
            # Process PDF
            print("\nProcessing PDF...")
            nodes = pdf_processor.process_pdf(config.pdf_path)
            print(f"Found {len(nodes)} sections in PDF")

            # Generate conversations
            print("\nGenerating conversations...")
            conversations = []
            for i, node in enumerate(nodes, 1):
                print(f"Processing section {i}/{len(nodes)}")
                conversation = conversation_generator.generate_conversation(
                    chunk=node.text,
                    is_first_segment=(i == 1)
                )
                conversations.append(conversation)

            # Save text conversations
            conversation_generator.save_conversations(conversations, config.text_output_path)
            print(f"\nText conversation saved to {config.text_output_path}")
            
            # Combine conversations for audio generation
            full_text = "".join(conversations)
        
        # Generate audio podcast in batches
        print("\nGenerating audio podcast...")
        audio_generator.generate_podcast(
            text=full_text,
            output_path=config.audio_output_path,
            batch_size=5  # Process 5 segments at a time
        )

    except Exception as e:
        print(f"\nError: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
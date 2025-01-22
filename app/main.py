from pathlib import Path
from pdf_processor import PDFProcessor
from conversation_generator import ConversationGenerator
from audio_generator import XTTSPodcastGenerator
from config import Config
import os
import torch
import time
import asyncio
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    start_time = time.time()
    try:
        logger.info("🚀 Initializing podcast generation...")
        config = Config()

        current_dir = Path.cwd()
        data_dir = current_dir / 'Data'
        pdf_path = data_dir / 'input.pdf'
        output_dir = data_dir / 'podcast_episodes'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📁 Current directory: {current_dir}")
        logger.info(f"📁 Data directory exists: {data_dir.exists()}")
        logger.info(f"📄 PDF exists: {pdf_path.exists()}")
        
        # Initialize XTTS generator
        use_gpu = torch.cuda.is_available()
        logger.info("🎵 Initializing XTTS2 generator...")
        audio_generator = XTTSPodcastGenerator(config, use_gpu=use_gpu)
        logger.info(f"💻 Using {'GPU' if use_gpu else 'CPU'} for audio generation")

        if os.path.exists(config.text_output_path):
            logger.info(f"📝 Found existing conversation at {config.text_output_path}")
            with open(config.text_output_path, 'r', encoding='utf-8') as f:
                full_text = f.read()
        else:
            pdf_processor = PDFProcessor()
            conversation_generator = ConversationGenerator(config.groq_api_key)
            
            logger.info("📚 Processing PDF...")
            nodes = pdf_processor.process_pdf(pdf_path)
            logger.info(f"📑 Found {len(nodes)} sections in PDF")

            logger.info("💭 Generating conversations...")
            conversations = []
            for i, node in enumerate(nodes, 1):
                logger.info(f"🔄 Processing section {i}/{len(nodes)}")
                conversation = await conversation_generator.generate_conversation_async(
                    chunk=node.text,
                    is_first_segment=(i == 1)
                )
                conversations.append(conversation)

            conversation_generator.save_conversations(conversations, config.text_output_path)
            full_text = "".join(conversations)
        
        logger.info("🎙️ Generating audio podcast...")
        output_path = output_dir / "podcast_output.mp3"
        audio_generator.generate_podcast(
            text=full_text,
            output_path=str(output_path)
        )

        execution_time = time.time() - start_time
        logger.info(f"✨ Completed in {execution_time:.2f} seconds")
        logger.info(f"📁 Output saved to: {output_path}")

    except Exception as e:
        logger.error(f"❌ Error: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(main())
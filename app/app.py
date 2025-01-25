import streamlit as st
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

async def process_pdf_and_generate(pdf_path, config, status_text, progress_bar):
    pdf_processor = PDFProcessor()
    conversation_generator = ConversationGenerator(config.groq_api_key)
    
    status_text.text("📚 Processing PDF...")
    nodes = pdf_processor.process_pdf(pdf_path)
    progress_bar.progress(30)
    
    status_text.text("💭 Generating conversations...")
    conversations = []
    for i, node in enumerate(nodes, 1):
        status_text.text(f"🔄 Processing section {i}/{len(nodes)}")
        conversation = await conversation_generator.generate_conversation_async(
            chunk=node.text,
            is_first_segment=(i == 1)
        )
        conversations.append(conversation)
        progress_bar.progress(30 + (40 * i // len(nodes)))

    conversation_generator.save_conversations(conversations, config.text_output_path)
    return "".join(conversations)

def main():
    st.title("PDF to Podcast Generator")
    
    config = Config()
    root_dir = Path(__file__).parent.parent
    data_dir = root_dir / 'Data'
    output_dir = data_dir / 'podcast_episodes'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    uploaded_file = st.file_uploader("Upload PDF file", type="pdf")
    
    if uploaded_file:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            start_time = time.time()
            temp_pdf_path = data_dir / "temp.pdf"
            with open(temp_pdf_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            status_text.text("🎵 Initializing XTTS2 generator...")
            use_gpu = torch.cuda.is_available()
            audio_generator = XTTSPodcastGenerator(config, use_gpu=use_gpu)
            progress_bar.progress(20)

            if os.path.exists(config.text_output_path):
                with open(config.text_output_path, 'r', encoding='utf-8') as f:
                    full_text = f.read()
            else:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                full_text = loop.run_until_complete(
                    process_pdf_and_generate(temp_pdf_path, config, status_text, progress_bar)
                )

            # Create two columns layout
            left_col, right_col = st.columns(2)
            
            with left_col:
                st.subheader("Generated Script")
                st.text_area("", full_text, height=600)
            
            with right_col:
                st.subheader("Generated Episodes")
                episode_slots = {}
            
            status_text.text("🎙️ Generating audio podcast...")
            output_path = output_dir / "podcast_output.mp3"
            audio_generator.generate_podcast(
                text=full_text,
                output_path=str(output_path)
            )
            progress_bar.progress(100)

            # Monitor and display episodes
            start_check_time = time.time()
            previous_episode_count = 0
            unchanged_count = 0

            while time.time() - start_check_time < 300:  # 5-minute timeout
                current_episodes = sorted(list(output_dir.glob("*.mp3")))
                
                # Check for new episodes
                for episode in current_episodes:
                    episode_path = str(episode)
                    if episode_path not in episode_slots:
                        with right_col:
                            episode_slots[episode_path] = st.empty()
                            episode_slots[episode_path].audio(episode_path)

                # Check if we're done
                if len(current_episodes) == previous_episode_count:
                    unchanged_count += 1
                    if unchanged_count >= 3:  # 30 seconds without changes
                        break
                else:
                    unchanged_count = 0
                    previous_episode_count = len(current_episodes)

                time.sleep(10)

            execution_time = time.time() - start_time
            status_text.text(f"✨ Completed in {execution_time:.2f} seconds")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            logger.error(f"❌ Error: {str(e)}", exc_info=True)
        
        finally:
            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)
    else:
        st.info("Please upload a PDF file to begin.")

if __name__ == "__main__":
    main()
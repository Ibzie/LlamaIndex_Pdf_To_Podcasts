from transformers import VitsModel, AutoTokenizer
import torch
import scipy.io.wavfile as wavfile
import re
from pydub import AudioSegment
import os
from typing import List, Tuple
import numpy as np
import warnings
from tqdm import tqdm

class AudioGenerator:
    def __init__(self, hf_token: str):
        warnings.filterwarnings("ignore")
        
        print("Loading MMS-TTS model...")
        model_name = "facebook/mms-tts-eng"
        self.model = VitsModel.from_pretrained(model_name, token=hf_token)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        print(f"Using device: {self.device}")

        # Different speakers for host and expert
        # Using English speakers: "en_US_1" through "en_US_8" are available
        self.host_voice = 0    # Male voice
        self.expert_voice = 1
        
        self.temp_dir = "temp_audio"
        os.makedirs(self.temp_dir, exist_ok=True)
        
        self.MAX_EPISODE_LENGTH = 5 * 60 * 1000  # 5 minutes in milliseconds
        self.SEGMENT_PAUSE = 500  # 500ms pause between segments

    def _split_conversation(self, text: str) -> List[Tuple[str, str]]:
        pattern = r'(Host|Expert):\s*((?:(?!Host:|Expert:).)*)'
        matches = re.finditer(pattern, text, re.DOTALL)
        
        segments = []
        for match in matches:
            speaker = match.group(1)
            dialogue = match.group(2).strip()
            if len(dialogue) > 0:
                segments.append((speaker, dialogue))
        
        return segments

    def _generate_audio_segment(self, text: str, is_host: bool, index: int) -> str:
        temp_path = os.path.join(self.temp_dir, f"segment_{index}.wav")
        
        try:
            # Select voice
            voice = self.host_voice if is_host else self.expert_voice
            
            # Tokenize text
            inputs = self.tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate speech
            with torch.no_grad():
                output = self.model(
                    **inputs,
                    speaker_id=torch.tensor([voice], device=self.device)
                )
                audio = output[0].cpu().numpy().squeeze()
            
            # Normalize and save
            audio = np.int16(audio * 32767)
            wavfile.write(temp_path, rate=self.model.config.sampling_rate, data=audio)
            
            return temp_path
            
        except Exception as e:
            print(f"\nError generating segment {index}: {str(e)}")
            return None

    def _create_episode_introduction(self, episode_num: int) -> str:
        intro_text = f"Welcome back to Part {episode_num} of our discussion."
        return self._generate_audio_segment(intro_text, is_host=True, index=f"intro_{episode_num}")

    def _create_episode_conclusion(self, episode_num: int) -> str:
        outro_text = f"This concludes Part {episode_num}. Please continue to the next part for more of our discussion."
        return self._generate_audio_segment(outro_text, is_host=True, index=f"outro_{episode_num}")

    def generate_podcast(self, text: str, output_path: str, batch_size: int = 5):
        try:
            print("\nAnalyzing conversation structure...")
            segments = self._split_conversation(text)
            total_segments = len(segments)
            print(f"Found {total_segments} segments to process")

            current_episode = 1
            current_episode_audio = AudioSegment.empty()
            current_batch_files = []
            
            episodes_dir = "podcast_episodes"
            os.makedirs(episodes_dir, exist_ok=True)

            print("\nGenerating audio segments...")
            for batch_start in range(0, total_segments, batch_size):
                batch_end = min(batch_start + batch_size, total_segments)
                print(f"\nProcessing batch {batch_start//batch_size + 1}/{(total_segments+batch_size-1)//batch_size}")
                
                for i in range(batch_start, batch_end):
                    speaker, text = segments[i]
                    print(f"  Generating segment {i+1}/{total_segments} ({speaker})")
                    
                    segment_path = self._generate_audio_segment(
                        text=text,
                        is_host=(speaker == "Host"),
                        index=i+1
                    )
                    
                    if segment_path:
                        current_batch_files.append(segment_path)
                        segment_audio = AudioSegment.from_wav(segment_path)
                        
                        # Check episode length
                        if len(current_episode_audio) + len(segment_audio) > self.MAX_EPISODE_LENGTH:
                            print(f"\nFinalizing Episode {current_episode}...")
                            
                            outro_path = self._create_episode_conclusion(current_episode)
                            if outro_path:
                                current_episode_audio += AudioSegment.from_wav(outro_path)
                            
                            episode_path = os.path.join(episodes_dir, f"episode_{current_episode}.mp3")
                            current_episode_audio.export(episode_path, format="mp3")
                            print(f"Saved Episode {current_episode}")
                            
                            current_episode += 1
                            current_episode_audio = AudioSegment.empty()
                            
                            intro_path = self._create_episode_introduction(current_episode)
                            if intro_path:
                                current_episode_audio = AudioSegment.from_wav(intro_path)
                        
                        if len(current_episode_audio) > 0:
                            current_episode_audio += AudioSegment.silent(duration=self.SEGMENT_PAUSE)
                        current_episode_audio += segment_audio
                
                # Clean up batch files
                for file in current_batch_files:
                    try:
                        os.remove(file)
                    except Exception as e:
                        print(f"Error removing temporary file {file}: {str(e)}")
                current_batch_files = []

            # Save final episode
            if len(current_episode_audio) > 0:
                outro_path = self._create_episode_conclusion(current_episode)
                if outro_path:
                    current_episode_audio += AudioSegment.from_wav(outro_path)
                
                episode_path = os.path.join(episodes_dir, f"episode_{current_episode}.mp3")
                current_episode_audio.export(episode_path, format="mp3")
                print(f"Saved final Episode {current_episode}")

            # Cleanup
            self.cleanup()
            print(f"\nGenerated {current_episode} episodes in: {episodes_dir}/")
            
        except Exception as e:
            print(f"\nError generating podcast: {str(e)}")
            self.cleanup()
            raise

    def cleanup(self):
        print("\nCleaning up temporary files...")
        if os.path.exists(self.temp_dir):
            for file in os.listdir(self.temp_dir):
                try:
                    os.remove(os.path.join(self.temp_dir, file))
                except Exception as e:
                    print(f"Error removing {file}: {str(e)}")
            os.rmdir(self.temp_dir)

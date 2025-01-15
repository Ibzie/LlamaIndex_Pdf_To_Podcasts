import torch
import re
from TTS.api import TTS
from pydub import AudioSegment
import os
from typing import List, Tuple
from tqdm import tqdm
from pathlib import Path


class AudioGenerator:
    def __init__(self, use_gpu: bool = True):
        print("\n🚀 Initializing Audio Generator...")
        
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        print(f"🖥️  Using device: {self.device}")
        
        print("\n📦 Loading TTS models...")
        try:
            # Male voice for host
            self.host_tts = TTS(model_name="tts_models/en/vctk/vits", progress_bar=True).to(self.device)
            # Female voice for expert
            self.expert_tts = TTS(model_name="tts_models/en/vctk/vits", progress_bar=True).to(self.device)
            
            # Select specific speakers from VCTK
            self.host_speaker = "p273"    # Male voice
            self.expert_speaker = "p299"  # Female voice
            
            print("✅ Models loaded successfully")
        except Exception as e:
            print(f"❌ Error loading models: {str(e)}")
            raise
        
        # Setup directories
        self.temp_dir = os.path.join(Path(__file__).parent.parent, "Data/temp_audio")
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Audio settings
        self.MAX_EPISODE_LENGTH = 10 * 60 * 1000  # 5 minutes
        self.SEGMENT_PAUSE = 500  # 500ms pause
        
        # Emotion settings through speed adjustment
        self.emotion_settings = {
            'neutral': 1.0,
            'excited': 1.1,
            'thoughtful': 0.95
        }

    def _detect_emotion(self, text: str) -> str:
        """Simple emotion detection based on text"""
        if '!' in text or any(word in text.lower() for word in ['excited', 'great', 'amazing']):
            return 'excited'
        elif '?' in text or any(word in text.lower() for word in ['think', 'perhaps', 'maybe']):
            return 'thoughtful'
        return 'neutral'

    def _split_conversation(self, text: str) -> List[Tuple[str, str, str]]:
        """Split conversation into segments"""
        pattern = r'(Host|T.E):\s*((?:(?!Host:|T.E:).)*)'
        matches = re.finditer(pattern, text, re.DOTALL)
        
        return [(m.group(1), m.group(2).strip(), self._detect_emotion(m.group(2))) 
                for m in matches if m.group(2).strip()]

    def _generate_audio_segment(self, text: str, is_host: bool, emotion: str, index: int) -> str:
        """Generate a single audio segment"""
        temp_path = os.path.join(self.temp_dir, f"segment_{index}.wav")
        
        try:
            tts_model = self.host_tts if is_host else self.expert_tts
            speaker = self.host_speaker if is_host else self.expert_speaker
            speed = self.emotion_settings[emotion]
            
            # Generate audio with selected speaker and emotion
            tts_model.tts_to_file(
                text=text,
                file_path=temp_path,
                speaker=speaker,
                speed=speed
            )
            return temp_path
            
        except Exception as e:
            print(f"\n❌ Error generating segment {index}: {str(e)}")
            return None

    def generate_podcast(self, text: str, output_path: str, batch_size: int = 3):
        """Generate full podcast"""
        try:
            print("\n📝 Analyzing conversation structure...")
            segments = self._split_conversation(text)
            total_segments = len(segments)
            print(f"📊 Found {total_segments} segments to process")

            current_episode = 1
            current_episode_audio = AudioSegment.empty()
            episodes_dir = os.path.join(Path(__file__).parent.parent, "Data/podcast_episodes")
            os.makedirs(episodes_dir, exist_ok=True)

            print("\n🎵 Generating audio segments...")
            total_batches = (total_segments + batch_size - 1) // batch_size
            
            for batch_start in range(0, total_segments, batch_size):
                batch_end = min(batch_start + batch_size, total_segments)
                print(f"\n🔄 Processing batch {batch_start//batch_size + 1}/{total_batches}")
                
                for i in tqdm(range(batch_start, batch_end), 
                            desc="Generating segments", 
                            total=batch_end-batch_start):
                    speaker, text, emotion = segments[i]
                    print(f"\n🎤 Segment {i+1}/{total_segments}")
                    print(f"👤 Speaker: {speaker}")
                    print(f"😊 Emotion: {emotion}")
                    
                    segment_path = self._generate_audio_segment(
                        text=text,
                        is_host=(speaker == "Host"),
                        emotion=emotion,
                        index=i+1
                    )
                    
                    if segment_path:
                        segment_audio = AudioSegment.from_wav(segment_path)
                        
                        if len(current_episode_audio) + len(segment_audio) > self.MAX_EPISODE_LENGTH:
                            episode_path = os.path.join(episodes_dir, f"episode_{current_episode}.mp3")
                            current_episode_audio.export(episode_path, format="mp3")
                            print(f"💿 Saved Episode {current_episode}")
                            
                            current_episode += 1
                            current_episode_audio = AudioSegment.empty()
                        
                        if len(current_episode_audio) > 0:
                            current_episode_audio += AudioSegment.silent(duration=self.SEGMENT_PAUSE)
                        current_episode_audio += segment_audio
                        
                        os.remove(segment_path)
                        print("✅ Segment processed successfully")

            if len(current_episode_audio) > 0:
                episode_path = os.path.join(episodes_dir, f"episode_{current_episode}.mp3")
                current_episode_audio.export(episode_path, format="mp3")
                print(f"💿 Saved final Episode {current_episode}")

            print(f"\n🎉 Generated {current_episode} episodes in: {episodes_dir}/")
            
        except Exception as e:
            print(f"\n❌ Error generating podcast: {str(e)}")
            raise
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up temporary files"""
        print("\n🧹 Cleaning up temporary files...")
        if os.path.exists(self.temp_dir):
            for file in os.listdir(self.temp_dir):
                try:
                    os.remove(os.path.join(self.temp_dir, file))
                except Exception as e:
                    print(f"⚠️ Error removing {file}: {str(e)}")
            os.rmdir(self.temp_dir)
            print("✨ Cleanup complete!")
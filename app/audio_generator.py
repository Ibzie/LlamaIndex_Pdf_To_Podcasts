import torch
import re
import numpy as np
from pathlib import Path
import concurrent.futures
from typing import List, Optional
from pydub import AudioSegment
import os
import random
from tqdm import tqdm
from TTS.api import TTS
import soundfile as sf
from config import Config

class XTTSPodcastGenerator:
    def __init__(self, config: Config, use_gpu: bool = True):
        print("\n🚀 Initializing XTTS2 Generator...")
        
        self.config = config
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
        
        root_dir = Path(__file__).parent.parent
        self.data_dir = root_dir / 'Data'
        self.reference_audio_path = self.data_dir / 'reference_voices'
        self.temp_dir = self.data_dir / 'temp_audio'
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        self._initialize_model()
        
        self.voices = {
            'host': str(self.reference_audio_path / "female_02.wav"),
            'expert': str(self.reference_audio_path / "male_01.wav")
        }
        
        for role, path in self.voices.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Voice file for {role} not found at {path}")
        
        self.MAX_CHUNK_SIZE = 15
        self.MAX_WORKERS = 1
        self.BATCH_SIZE = 1
        self.MAX_EPISODE_LENGTH = 1 * 60 * 1000
        
        self._conversation_pattern = re.compile(
            r'(Host|T\.E):\s*(?:\[[\w\s]+\])?\s*((?:(?!Host:|T\.E:).)*)',
            re.DOTALL
        )
        
        self._setup_voice_patterns()

    def _initialize_model(self):
        model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        self.model = TTS(model_name).to(self.device)
        torch.set_grad_enabled(False)

    def _setup_voice_patterns(self):
        self.voice_settings = {
            'neutral': {'speed': 1.0},
            'excited': {'speed': 1.2},
            'thoughtful': {'speed': 0.9},
            'serious': {'speed': 1.0},
            'confident': {'speed': 1.1}
        }

    def _detect_emotion(self, text: str, is_host: bool) -> str:
        text_lower = text.lower()
        if '!' in text:
            return 'excited'
        elif '?' in text:
            return 'thoughtful'
        elif any(word in text_lower for word in ['must', 'should', 'will']):
            return 'serious'
        elif any(word in text_lower for word in ['absolutely', 'certainly']):
            return 'confident'
        return 'neutral'

    def _optimize_text(self, text: str) -> List[str]:
        sentences = re.split('[.!?]+', text)
        chunks = [s.strip() for s in sentences if s.strip()]
        return chunks

    def _generate_audio_chunk(self, text: str, voice_path: str, emotion: str) -> Optional[np.ndarray]:
        try:
            wav = self.model.tts(
                text=text,
                speaker_wav=voice_path,
                language="en"
            )
            return wav
        except Exception as e:
            print(f"❌ Error generating chunk: {str(e)}")
            return None

    def _process_segment(self, text: str, is_host: bool, emotion: str, index: int) -> Optional[str]:
        try:
            chunks = self._optimize_text(text)
            voice_path = self.voices['host'] if is_host else self.voices['expert']
            
            all_audio = []
            for chunk in chunks:
                if len(chunk) < 3:
                    continue
                audio_array = self._generate_audio_chunk(chunk, voice_path, emotion)
                if audio_array is not None:
                    all_audio.append(audio_array)
            
            if not all_audio:
                return None
            
            combined_audio = np.concatenate([
                np.concatenate([chunk, np.zeros(int(22050 * 0.2))])
                for chunk in all_audio
            ])
            
            output_path = self.temp_dir / f"segment_{index}.wav"
            sf.write(str(output_path), combined_audio, 22050)
            return str(output_path)
            
        except Exception as e:
            print(f"❌ Error processing segment: {str(e)}")
            return None

    def generate_podcast(self, text: str, output_path: str):
        try:
            matches = self._conversation_pattern.finditer(text)
            segments = []
            for m in matches:
                if m.group(2).strip():
                    text = re.sub(r'\[.*?\]', '', m.group(2)).strip()  # Remove emotion tags
                    segments.append((m.group(1), ' '.join(text.split())))
            
            print(f"📊 Processing {len(segments)} segments")
            current_episode = 1
            current_audio = AudioSegment.empty()
            episodes_dir = Path(output_path).parent
            episodes_dir.mkdir(parents=True, exist_ok=True)
            
            for i, (speaker, text) in enumerate(tqdm(segments)):
                is_host = speaker == "Host"
                emotion = self._detect_emotion(text, is_host)
                
                print(f"\n🎤 Processing: {speaker}")
                print(f"😊 Emotion detected: {emotion}")
                
                segment_path = self._process_segment(
                    text=text,
                    is_host=is_host,
                    emotion=emotion,
                    index=i
                )
                
                if segment_path:
                    segment_audio = AudioSegment.from_wav(segment_path)
                    
                    if len(current_audio) + len(segment_audio) > self.MAX_EPISODE_LENGTH:
                        episode_path = episodes_dir / f"episode_{current_episode}.mp3"
                        current_audio.export(
                            str(episode_path),
                            format="mp3",
                            parameters=["-q:a", "2"]
                        )
                        print(f"💿 Saved Episode {current_episode}")
                        current_episode += 1
                        current_audio = AudioSegment.empty()
                    
                    if len(current_audio) > 0:
                        current_audio += AudioSegment.silent(duration=250)
                    
                    current_audio += segment_audio
                    os.remove(segment_path)
            
            if len(current_audio) > 0:
                episode_path = episodes_dir / f"episode_{current_episode}.mp3"
                current_audio.export(
                    str(episode_path),
                    format="mp3",
                    parameters=["-q:a", "2"]
                )
                print(f"💿 Saved final Episode {current_episode}")
            
        except Exception as e:
            print(f"\n❌ Error generating podcast: {str(e)}")
            raise
        finally:
            self.cleanup()

    def cleanup(self):
        if self.temp_dir.exists():
            for file in self.temp_dir.glob("*.wav"):
                try:
                    file.unlink()
                except Exception as e:
                    print(f"⚠️ Error removing {file}: {str(e)}")
            self.temp_dir.rmdir()
            print("\n🧹 Cleanup complete!")
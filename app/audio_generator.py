import torch
import re
from TTS.api import TTS
from pydub import AudioSegment
import os
from typing import List, Tuple
from tqdm import tqdm
from pathlib import Path
import random
from config import Config
import concurrent.futures
import numpy as np

class TTSModelCache:
    _instance = None
    _voice_refs = {}
    
    @classmethod
    def get_instance(cls, device: str):
        if cls._instance is None:
            model_path = os.path.expanduser('~/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/')
            if not os.path.exists(model_path):
                print("\n⚠️ Model not found locally. Downloading (this may take a while)...")
            else:
                print("\n✅ Using cached model from:", model_path)
            
            try:
                cls._instance = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
                print("✅ Model loaded successfully")
            except Exception as e:
                print(f"❌ Error loading model: {str(e)}")
                raise
        return cls._instance
    
    @classmethod
    def get_voice_ref(cls, speaker: str, ref_audio_path: str):
        if speaker not in cls._voice_refs:
            print(f"\n🎤 Loading voice reference for {speaker}...")
            try:
                audio_path = os.path.join(ref_audio_path, f"{speaker}.wav")
                if not os.path.exists(audio_path):
                    raise FileNotFoundError(f"Voice reference file not found: {audio_path}")
                cls._voice_refs[speaker] = open(audio_path, 'rb').read()
                print("✅ Voice reference loaded successfully")
            except Exception as e:
                print(f"❌ Error loading voice reference: {str(e)}")
                raise
        return cls._voice_refs[speaker]

class AudioGenerator:
    def __init__(self, use_gpu: bool = True):
        self.config = Config()
        print("\n🚀 Initializing Audio Generator...")
        
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            torch.cuda.empty_cache()
        print(f"🖥️  Using device: {self.device}")
        
        self.tts_model = TTSModelCache.get_instance(self.device)
        self.host_speaker = "female_02"
        self.expert_speaker = "male_01"
        
        self.temp_dir = os.path.join(Path(__file__).parent.parent, "Data/temp_audio")
        os.makedirs(self.temp_dir, exist_ok=True)
        
        self.MAX_EPISODE_LENGTH = 1 * 60 * 1000
        self.MAX_WORKERS = 1 if self.device == "cuda" else 2

        self.emotion_settings = {
            'neutral': {'speed': 1.0},
            'excited': {'speed': 1.15},
            'thoughtful': {'speed': 0.92},
            'serious': {'speed': 0.95},
            'happy': {'speed': 1.08},
            'confident': {'speed': 1.05},
            'skeptical': {'speed': 0.95},
            'analytical': {'speed': 0.97},
            'emphatic': {'speed': 1.1},
            'curious': {'speed': 1.02},
            'authoritative': {'speed': 0.98}
        }

        self.pause_variations = {
            '.': (600, 800),
            ',': (200, 300),
            '...': (400, 600),
            '?': (500, 700),
            '!': (500, 700),
            ':': (300, 400)
        }

    def _normalize_text(self, text: str) -> str:
        text = ' '.join(text.split())
        if text and text[-1] not in '.!?':
            text += '.'
        return text

    def _detect_emotion(self, text: str) -> str:
        text_lower = text.lower()
        word_count = len(text.split())
        
        questions = text.count('?')
        exclamations = text.count('!')
        
        patterns = {
            'analytical': (r'\b(?:analyze|calculate|measure|evaluate)\b', 3),
            'authoritative': (r'\b(?:must|should|will|always|never)\b', 2),
            'curious': (r'\b(?:how|why|what if|could|would)\b', 2)
        }
        
        for emotion, (pattern, threshold) in patterns.items():
            if len(re.findall(pattern, text_lower)) >= threshold:
                return emotion
                
        if exclamations > 1:
            return 'excited'
        elif questions > 1:
            return 'thoughtful'
        elif word_count > 30:
            return 'serious'
        
        return 'neutral'

    def _split_conversation(self, text: str) -> List[Tuple[str, str, str]]:
        try:
            pattern = r'(Host Rachel|T\.E \(Kevin\)):\s*((?:(?!Host Rachel:|T\.E \(Kevin\):).)*)'
            matches = re.finditer(pattern, text, re.DOTALL)
            segments = []
            
            for m in matches:
                if m.group(2).strip():
                    text = self._normalize_text(m.group(2).strip())
                    emotion = self._detect_emotion(text)
                    segments.append((m.group(1), text, emotion))
            
            return segments
        except Exception as e:
            print(f"❌ Error splitting conversation: {str(e)}")
            raise

    def _add_natural_pauses(self, audio: AudioSegment, text: str) -> AudioSegment:
        try:
            final_audio = AudioSegment.empty()
            current_pos = 0
            
            for char, pause_range in self.pause_variations.items():
                positions = [i for i in range(len(text)) if text.startswith(char, i)]
                for pos in positions:
                    if pos > current_pos:
                        final_audio += audio[current_pos*100:pos*100]
                        final_audio += AudioSegment.silent(duration=random.randint(*pause_range))
                        current_pos = pos + 1
            
            if current_pos < len(text):
                final_audio += audio[current_pos*100:]
                
            return final_audio
        except Exception as e:
            print(f"❌ Error adding natural pauses: {str(e)}")
            return audio

    def _chunk_text(self, text: str, max_length: int = 250) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if sentence_length > max_length:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                sub_chunks = []
                remaining = sentence
                while len(remaining) > max_length:
                    split_point = max_length
                    for punct in [',', ';', ' ']:
                        last_punct = remaining[:max_length].rfind(punct)
                        if last_punct != -1:
                            split_point = last_punct + 1
                            break
                    
                    sub_chunks.append(remaining[:split_point].strip())
                    remaining = remaining[split_point:].strip()
                
                if remaining:
                    sub_chunks.append(remaining)
                
                chunks.extend(sub_chunks)
                continue
                
            if current_length + sentence_length <= max_length:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

    def _generate_single_audio_segment(self, text: str, is_host: bool, emotion: str, index: str) -> str:
        temp_path = os.path.join(self.temp_dir, f"segment_{index}.wav")
        
        try:
            speaker = self.host_speaker if is_host else self.expert_speaker
            settings = self.emotion_settings.get(emotion, {'speed': 1.0})
            voice_ref = TTSModelCache.get_voice_ref(speaker, self.config.refrence_audio_path)
            
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            self.tts_model.tts_to_file(
                text=text,
                file_path=temp_path,
                speaker_wav=voice_ref,
                language="en",
                speed=settings['speed']
            )
            
            audio = AudioSegment.from_wav(temp_path)
            audio = self._add_natural_pauses(audio, text)
            audio.export(temp_path, format='wav')
            
            return temp_path
            
        except Exception as e:
            print(f"\n❌ Error generating segment {index}: {str(e)}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return None

    def _generate_audio_segment(self, text: str, is_host: bool, emotion: str, index: int) -> str:
        chunks = self._chunk_text(text)
        if len(chunks) == 1:
            return self._generate_single_audio_segment(chunks[0], is_host, emotion, index)
        
        temp_chunks = []
        merged_audio = AudioSegment.empty()
        
        for i, chunk in enumerate(chunks):
            chunk_path = self._generate_single_audio_segment(chunk, is_host, emotion, f"{index}_{i}")
            if chunk_path:
                temp_chunks.append(chunk_path)
                chunk_audio = AudioSegment.from_wav(chunk_path)
                if len(merged_audio) > 0:
                    merged_audio += AudioSegment.silent(duration=100)
                merged_audio += chunk_audio
        
        final_path = os.path.join(self.temp_dir, f"segment_{index}.wav")
        merged_audio.export(final_path, format='wav')
        
        for chunk_path in temp_chunks:
            os.remove(chunk_path)
        
        return final_path

    def generate_podcast(self, text: str, output_path: str, batch_size: int = 1):
        try:
            segments = self._split_conversation(text)
            total_segments = len(segments)
            print(f"📊 Found {total_segments} segments to process")

            current_episode = 1
            current_episode_audio = AudioSegment.empty()
            episodes_dir = os.path.join(Path(__file__).parent.parent, "Data/podcast_episodes")
            os.makedirs(episodes_dir, exist_ok=True)

            print("\n🎵 Generating audio segments...")
            
            for i, (speaker, text, emotion) in enumerate(segments):
                print(f"\n🎤 Processing segment {i+1}/{total_segments}")
                print(f"👤 Speaker: {speaker}")
                print(f"😊 Emotion: {emotion}")
                
                segment_path = self._generate_audio_segment(
                    text=text,
                    is_host=(speaker == "Host Rachel"),
                    emotion=emotion,
                    index=i
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
                        current_episode_audio += AudioSegment.silent(duration=random.randint(400, 600))
                    
                    current_episode_audio += segment_audio
                    os.remove(segment_path)
                    print("✅ Segment processed successfully")

            if len(current_episode_audio) > 0:
                episode_path = os.path.join(episodes_dir, f"episode_{current_episode}.mp3")
                current_episode_audio.export(episode_path, format="mp3")
                print(f"💿 Saved final Episode {current_episode}")

        except Exception as e:
            print(f"\n❌ Error generating podcast: {str(e)}")
            raise
        finally:
            self.cleanup()

    def cleanup(self):
        print("\n🧹 Cleaning up temporary files...")
        if os.path.exists(self.temp_dir):
            for file in os.listdir(self.temp_dir):
                try:
                    os.remove(os.path.join(self.temp_dir, file))
                except Exception as e:
                    print(f"⚠️ Error removing {file}: {str(e)}")
            os.rmdir(self.temp_dir)
            print("✨ Cleanup complete!")
from groq import AsyncGroq
from typing import List
import os
import asyncio
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Dict, Any

class ConversationGenerator:
    def __init__(self, api_key: str, max_history: int = 5, max_workers: int = 3):
        self.client = AsyncGroq(api_key=api_key)
        self.conversation_history = []
        self.max_history = max_history
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    @staticmethod
    @lru_cache(maxsize=128)
    def _get_system_prompt() -> str:
        return """You are generating a podcast conversation between a Host (Named Rachel) and an Expert(Named Kevin). The conversation should dynamically adapt tone and emotion based on context:

Key Features:
- Utilize varied emotional states: analytical, confident, skeptical, emphatic, curious, authoritative
- Match tone to content: technical explanations → analytical, challenging ideas → skeptical, groundbreaking findings → excited
- Naturally transition between emotions based on topic shifts
- Include emotional indicators in speech patterns (e.g., "Fascinating..." for curious, "Let me demonstrate..." for confident)

Conversation Guidelines:
- Build natural flow with appropriate emotional progression
- Incorporate technical markers when discussing research/data
- Use questioning patterns to show curiosity/skepticism
- Include authoritative language when presenting expert opinions
- Maintain emotional coherence across topic transitions
- Reference previous points with appropriate emotional callback

Structure:
- Continue existing conversation flow
- Skip introductions after first segment
- Use context-appropriate emotional transitions
- Keep technical accuracy while varying emotional delivery
- Include reactive elements matching emotional context

Format:
Host: [Context-driven emotional response]
T.E: [Expertise-based emotional delivery]"""

    def append_history(self, conversation: str):
        self.conversation_history.append(conversation)
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)

    async def generate_conversation_async(self, chunk: str, is_first_segment: bool) -> str:
        try:
            context = "This is the first segment. Start with brief introductions." if is_first_segment else "Continue the ongoing conversation naturally."
            
            messages = [
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": f"{context}\n\nContent: {chunk}"}
            ]
            
            if self.conversation_history:
                messages.insert(1, {"role": "assistant", "content": self.conversation_history[-1]})

            response = await self.client.chat.completions.create(
                messages=messages,
                model="llama-3.1-8b-instant",
                temperature=0.7,
                max_tokens=4096
            )
            
            conversation = response.choices[0].message.content
            self.append_history(conversation)
            return conversation
            
        except Exception as e:
            raise Exception(f"Error generating conversation: {str(e)}")

    async def process_chunks(self, chunks: List[str]) -> List[str]:
        tasks = []
        for i, chunk in enumerate(chunks):
            tasks.append(self.generate_conversation_async(chunk, i == 0))
        return await asyncio.gather(*tasks)

    def save_conversations(self, conversations: List[str], output_path: str, batch_size: int = 1000):
        try:
            with open(output_path, 'w', encoding='utf-8', buffering=8192) as f:
                for i in range(0, len(conversations), batch_size):
                    batch = conversations[i:i + batch_size]
                    f.writelines(f"{conv}\n\n" for conv in batch)
        except Exception as e:
            raise Exception(f"Error saving conversations: {str(e)}")

async def main(api_key: str, chunks: List[str], output_path: str):
    generator = ConversationGenerator(api_key)
    conversations = await generator.process_chunks(chunks)
    generator.save_conversations(conversations, output_path)

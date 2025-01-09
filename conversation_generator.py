from groq import Groq
from typing import List
import os

class ConversationGenerator:
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        self.conversation_history = []

    @staticmethod
    def _get_system_prompt() -> str:
        return """You are generating a podcast conversation between a Host and an Expert. Continue the existing conversation naturally, maintaining context and flow. Don't restart introductions or topics already covered. Use casual language while keeping technical accuracy. Include reactions and follow-ups that build on previous segments.

Rules:
- Continue the conversation flow from previous segments
- Only introduce speakers in the very first segment
- Reference previously discussed points when relevant
- Use natural transitions between topics
- Keep the casual, engaging tone throughout"""

    def generate_conversation(self, chunk: str, is_first_segment: bool) -> str:
        try:
            context = "This is the first segment. Start with brief introductions." if is_first_segment else "Continue the ongoing conversation naturally."
            
            messages = [
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": f"{context}\n\nContent: {chunk}"}
            ]
            
            # Add conversation history for context
            if self.conversation_history:
                messages.insert(1, {"role": "assistant", "content": self.conversation_history[-1]})

            response = self.client.chat.completions.create(
                messages=messages,
                model="mixtral-8x7b-32768",
                temperature=0.7,
                max_tokens=4096
            )
            
            conversation = response.choices[0].message.content
            self.conversation_history.append(conversation)
            return conversation
        except Exception as e:
            raise Exception(f"Error generating conversation: {str(e)}")

    def save_conversations(self, conversations: List[str], output_path: str):
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for conv in conversations:
                    f.write(conv)
                    f.write("\n\n")
        except Exception as e:
            raise Exception(f"Error saving conversations: {str(e)}")
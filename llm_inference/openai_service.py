import os
from openai import OpenAI

from .llm_interface import LLMInterface

class OpenAIService(LLMInterface):
    def __init__(self, model_name, temperature, max_tokens):
        super().__init__(model_name, temperature, max_tokens)
        self.client = OpenAI(api_key=os.environ.get("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"))

    def create_prompt(self, system_prompt, few_shot_examples, question):
        messages = []
        messages.append({"role": "system", "content": system_prompt})
        for message in few_shot_examples:
            messages.append({"role": "user", "content": message['question']})
            messages.append({"role": "assistant", "content": message['answer']})
        messages.append({"role": "user", "content": question})
        return messages

    def make_request(self, messages):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content

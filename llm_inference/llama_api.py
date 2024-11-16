import os
import requests

class LLaMAService:
    def __init__(self, model_name, temperature, max_tokens):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = os.getenv("LLAMA_API_KEY", "c36df7f0-1253-4e97-8a36-b6603a35c473")  # Dùng API key từ môi trường

    def create_prompt(self, system_prompt, few_shot_examples, question):
        messages = [{"role": "system", "content": system_prompt}]
        for example in few_shot_examples:
            messages.append({"role": "user", "content": example['question']})
            messages.append({"role": "assistant", "content": example['answer']})
        messages.append({"role": "user", "content": question})
        return messages

    def make_request(self, messages):
        # Cấu hình URL API (đọc từ tài liệu LLaMA API)
        url = "https://api.llamaapi.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }

        # Gửi yêu cầu POST tới API
        response = requests.post(url, json=payload, headers=headers)

        # Xử lý phản hồi
        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        else:
            raise Exception(f"Error {response.status_code}: {response.text}")

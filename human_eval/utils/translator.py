import json
from ollama import chat, ChatResponse


class Translator:
    def __init__(self):
        config = json.load(open("config.json", "r"))

        self.translation_model = config["translation_model"]
        self.languages = config["languages"]

    def _translate_prompt(self, target_language: str, prompt: str) -> str:
        prompt = f"Translate the following progamming problem from english into {target_language}:\n{prompt}"

        response: ChatResponse = chat(
            model=self.translation_model, messages=[{"role": "user", "content": prompt}]
        )

        return response["response"]["content"]

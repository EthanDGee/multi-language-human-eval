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

    def _translate_problem(self, problem: str) -> dict:
        translated_problems = {"english": problem}

        for lang in self.languages:
            translated_problems[lang] = self._translate_prompt(lang, problem)

        return translated_problems

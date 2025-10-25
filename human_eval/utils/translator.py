import json
from ollama import chat, ChatResponse
from ..data import read_problems, write_jsonl


class Translator:
    def __init__(self):
        config = json.load(open("human_eval/utils/config.json", "r"))

        self.translation_model = config["translation_model"]
        self.languages = config["languages"]

    def _translate_prompt(self, target_language: str, prompt: str) -> str:
        prompt = f"Translate the following progamming problem from english into {target_language}:\n{prompt}"

        response: ChatResponse = chat(
            model=self.translation_model, messages=[{"role": "user", "content": prompt}]
        )

        return response["response"]["content"]

    def _translate_problem(self, problem: str) -> dict:
        problem_translations = {"english": problem}

        for lang in self.languages:
            problem_translations[lang] = self._translate_prompt(lang, problem)

        return problem_translations

    def translate_dataset(self):
        problems = read_problems()

        translated_prompts = []
        for i, task_id in enumerate(problems):
            prob = problems[task_id]["prompt"]

            translated_prompts.append(self._translate_problem(prob))
            translated_prompts[i]["task_id"] = task_id

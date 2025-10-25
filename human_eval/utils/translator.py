import json
from ollama import chat, ChatResponse
from ..data import read_problems, write_jsonl


class Translator:
    def __init__(self):
        config = json.load(open("human_eval/utils/config.json", "r"))

        self.translation_model = config["translation_model"]
        self.languages = config["languages"]

    def _translate_prompt(self, target_language: str, prompt: str) -> str:
        """
        Translates a given programming problem prompt from English into the specified target language by
        using the specified translation mode through the ollama API

        :param target_language: The language into which the prompt should be translated.
        :type target_language: str
        :param prompt: The programming problem prompt to translate.
        :type prompt: str
        :return: The translated programming problem prompt in the specified target language.
        :rtype: str
        """
        prompt = f"Translate the following progamming problem from english into {target_language}:\n{prompt}"

        response: ChatResponse = chat(
            model=self.translation_model, messages=[{"role": "user", "content": prompt}]
        )

        return response["response"]["content"]

    def _translate_problem(self, problem: str) -> dict:
        """
        Translates a given problem statement into multiple languages based on self.languages.
        The translation is stored in a dictionary where the key is the language and the
        value is the translated string. The source English problem is always included.

        :param problem: The problem statement in English to be translated.
        :type problem: str
        :return: A dictionary where keys represent the language and values are the translated
            problem statement.
        :rtype: dict
        """
        problem_translations = {"english": problem}

        for lang in self.languages:
            problem_translations[lang] = self._translate_prompt(lang, problem)

        return problem_translations

    def translate_dataset(self):
        """
        Translates the dataset of problems by processing each problem's prompt into self.languages.

        This function reads a set of problems, translates their prompts using an internal translation
        method, and collects the results in a list. The translated prompts are also assigned their
        respective task IDs for reference.

        :return: A list of translated prompts, where each prompt includes its respective task ID
        :rtype: list
        """
        problems = read_problems()

        translated_prompts = []
        for i, task_id in enumerate(problems):
            prob = problems[task_id]["prompt"]

            translated_prompts.append(self._translate_problem(prob))
            translated_prompts[i]["task_id"] = task_id

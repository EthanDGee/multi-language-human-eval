import json
import time
import warnings
from copy import deepcopy
from ..data import read_problems, write_jsonl
from ollama import chat, ChatResponse


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
        prompt = f"""Translate the following problem description and examples from english 
        into {target_language} DO NOT SOLVE THE PROBLEM ONLY OUTPUT THE PROBLEM DESCRIPTION
        + EXAMPLES IN THE TARGET LANGUAGE ANY GENERATED CODE IS A FAILED RESULT ONLY OUTPUT
        TRANSLATED TEXT THEN STOP GENERATION:\n{prompt}"""

        return self._call_ollama_with_retry(prompt)
    
    def _call_ollama_with_retry(self, prompt: str, max_retries=3, retry_delay=5) -> str:
        """
        Call ollama API with retry mechanism and error handling.
        
        :param prompt: The prompt to send to ollama
        :param max_retries: Maximum number of retry attempts
        :param retry_delay: Delay between retries in seconds
        :return: The response content from ollama
        """
        for attempt in range(max_retries):
            try:
                response: ChatResponse = chat(
                    model=self.translation_model, 
                    messages=[{"role": "user", "content": prompt}]
                )
                
                # Validate response structure
                if not response or "message" not in response or "content" not in response["message"]:
                    raise ValueError("Invalid response structure from ollama API")
                
                content = response["message"]["content"]
                if not content or not content.strip():
                    raise ValueError("Empty response content from ollama API")
                
                return content
                
            except ConnectionError as e:
                warnings.warn(f"Connection error to ollama API. Attempt {attempt + 1}/{max_retries}. Error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                raise ConnectionError(f"Failed to connect to ollama API after {max_retries} attempts: {e}")
                
            except TimeoutError as e:
                warnings.warn(f"Timeout error from ollama API. Attempt {attempt + 1}/{max_retries}. Error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                raise TimeoutError(f"Ollama API timeout after {max_retries} attempts: {e}")
                
            except ValueError as e:
                warnings.warn(f"Invalid response from ollama API. Attempt {attempt + 1}/{max_retries}. Error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                raise ValueError(f"Invalid ollama API response after {max_retries} attempts: {e}")
                
            except Exception as e:
                warnings.warn(f"Unexpected error from ollama API. Attempt {attempt + 1}/{max_retries}. Error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                raise Exception(f"Failed to call ollama API after {max_retries} attempts: {e}")

    def _translate_problem(self, problem: str, existing_translations: dict | None = None) -> dict:
        """
        Translates a given problem statement into multiple languages based on self.languages.
        The translation is stored in a dictionary where the key is the language and the
        value is the translated string. The source English problem is always included.
        Skips languages that are already translated.

        :param problem: The problem statement in English to be translated.
        :type problem: str
        :param existing_translations: Existing translations to skip, defaults to None
        :type existing_translations: dict, optional
        :return: A dictionary where keys represent the language and values are the translated
            problem statement.
        :rtype: dict
        """
        problem_translations = {"english": problem}
        if existing_translations:
            problem_translations.update(existing_translations)
        
        print("Target Problem")
        print(problem_translations)

        for lang in self.languages:
            if lang in problem_translations:
                print(f"Skipping {lang} - already translated")
                continue
                
            problem_translations[lang] = self._translate_prompt(lang, problem)

            print(f"Target Language: {lang}")
            print(problem_translations[lang])
            print()

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
        for problem_num, task_id in enumerate(problems):
            print(f"Problem {problem_num + 1}/164")
            entry = deepcopy(problems[task_id])
            entry["task_id"] = task_id

            prob = entry.pop("prompt")
            entry["prompts"] = self._translate_problem(prob)

            translated_prompts.append(entry)

        write_jsonl("data/TranslatedHumanEval.jsonl.gz", translated_prompts)


def main():
    translator = Translator()
    translator.translate_dataset()


if __name__ == "__main__":
    main()

## Multi-lingual HumanEval

This project is a fork of the original OpenAI HumanEval, expanded to include support for multiple human languages. This allows for the evaluation of code generation models in contexts beyond English.

The currently supported languages are: Mandarin Chinese, Spanish, Hindi, Portuguese, Bengali, French, Russian, Arabic, Japanese, and Punjabi. These translations were generated using the `mistral:7b-instruct` model.

### Adding New Languages

To add more languages, you can modify the `human_eval/utils/config.json` file to include the new language. You would then need to translate the problems into the new language and ensure the evaluation harness can correctly process them.

The process involves three main steps:

1. **Configuration**: Specifying the new language.
2. **Translation**: Generating the translated problem set.
3. **Evaluation**: Adapting the evaluation script to use the translated problems for a specific language.

#### Step 1: Configuration (`human_eval/utils/config.json`)

1. **Open `human_eval/utils/config.json`**.
2. **Add the language name** to the `languages` array. The name you use here (e.g., `"german"`, `"korean"`) will be used to request translations from the model and as a key to access the translated prompts.

    *Example `config.json` with German added:*

    ```json
    {
      "languages": [
        "mandarin-chinese",
        "spanish",
        "hindi",
        "portugese",
        "bengali",
        "french",
        "russian",
        "arabic",
        "japanese",
        "punjabi",
        "german"
      ],
      "translation_model": "mistral:7b-instruct"
    }
    ```

#### Step 2: Translation (`human_eval/utils/translator.py`)

1. **Ensure Ollama is running**: The `translator.py` script relies on a local Ollama instance to perform the translations using the specified `translation_model`. Make sure Ollama is installed and running, and that the `translation_model` has been pulled and is available locally (e.g., by running `ollama pull zongwei/gemma3-translator:4b`).

2. **Run the translation script**: Execute the following command from the project root:

    ```bash
    python -m  human_eval.utils.translator
    ```

    This script will save the comprehensive translated problem set to `data/TranslatedHumanEval.jsonl.gz`. Each entry in this file will now contain an additional field, `"prompts"`, which is a dictionary that maps language names to their respective translated problem descriptions.

## Installation

Check out and install this repository:

```bash
git clone <YOUR_GIT_REPOSITORY_URL>
cd multi-language-human-eval
```

Make sure to use python 3.10 or later:

```bash
python -m venv .venv
source .venv/bin/activate 
pip install -e .
```

## Usage

**This program exists to run untrusted model-generated code. Users are strongly
encouraged not to do so outside of a robust security sandbox. The [execution
call](https://github.com/openai/human-eval/blob/master/human_eval/execution.py#L48-L58)
in `execution.py` is deliberately commented out to ensure users read this
disclaimer before running code in a potentially unsafe manner. See the comment in
`execution.py` for more information and instructions.**

After following the above instructions to enable execution, generate samples
and save them in the following JSON Lines (jsonl) format, where each sample is
formatted into a single line like so:

```json
{"task_id": "Corresponding HumanEval task ID", "completion": "Completion only without the prompt"}
```

We provide `example_problem.jsonl` and `example_solutions.jsonl` under `data`
to illustrate the format and help with debugging.

Here is nearly functional example code (you just have to provide
`generate_one_completion` to make it work) that saves generated completions to
`samples.jsonl`.

```python
from human_eval.data import write_jsonl, read_problems

problems = read_problems()

num_samples_per_task = 200
samples = [
    dict(task_id=task_id, completion=generate_one_completion(problems[task_id]["prompt"]))
    for task_id in problems
    for _ in range(num_samples_per_task)
]
write_jsonl("samples.jsonl", samples)
```

### Evaluating in a Specific Language

To evaluate the samples in a language other than English, you can now use the `--language` argument with `evaluate_functional_correctness`. The `TranslatedHumanEval.jsonl.gz` file is now the default problem file, but you can override it with `--problem_file` if needed.

Example for evaluating in Spanish:

```bash
evaluate_functional_correctness samples.jsonl --language spanish
```

Or, if you need to specify a different problem file:

```bash
evaluate_functional_correctness samples.jsonl --problem_file data/MyCustomTranslatedEval.jsonl.gz --language german
```

As a quick sanity-check, the example samples should yield 0.5 pass@1.

```bash
$ evaluate_functional_correctness data/example_samples.jsonl --problem_file=data/example_problem.jsonl
Reading samples...
6it [00:00, 3397.11it/s]
Running example suites...
100%|...| 6/6 [00:03<00:00,  1.96it/s]
Writing results to data/example_samples.jsonl_results.jsonl...
100%|...| 6/6 [00:00<00:00, 6148.50it/s]
{'pass@1': 0.4999999999999999}
```

Because there is no unbiased way of estimating pass@k when there are fewer
samples than k, the script does not evaluate pass@k for these cases. To
evaluate with other k values, pass `--k=<comma-separated-values-here>`. For
other options, see

```bash
evaluate_functional_correctness --help
```

However, we recommend that you use the default values for the rest.

## Known Issues

While evaluation uses very little memory, you might see the following error
message when the system is running out of RAM. Since this may cause some
correct programs to fail, we recommend that you free some memory and try again.

```
malloc: can't allocate region
```

## Citation

```
@article{chen2021codex,
  title={Evaluating Large Language Models Trained on Code},
  author={Mark Chen and Jerry Tworek and Heewoo Jun and Qiming Yuan and Henrique Ponde de Oliveira Pinto and Jared Kaplan and Harri Edwards and Yuri Burda and Nicholas Joseph and Greg Brockman and Alex Ray and Raul Puri and Gretchen Krueger and Michael Petrov and Heidy Khlaaf and Girish Sastry and Pamela Mishkin and Brooke Chan and Scott Gray and Nick Ryder and Mikhail Pavlov and Alethea Power and Lukasz Kaiser and Mohammad Bavarian and Clemens Winter and Philippe Tillet and Felipe Petroski Such and Dave Cummings and Matthias Plappert and Fotios Chantzis and Elizabeth Barnes and Ariel Herbert-Voss and William Hebgen Guss and Alex Nichol and Alex Paino and Nikolas Tezak and Jie Tang and Igor Babuschkin and Suchir Balaji and Shantanu Jain and William Saunders and Christopher Hesse and Andrew N. Carr and Jan Leike and Josh Achiam and Vedant Misra and Evan Morikawa and Alec Radford and Matthew Knight and Miles Brundage and Mira Murati and Katie Mayer and Peter Welinder and Bob McGrew and Dario Amodei and Sam McCandlish and Ilya Sutskever and Wojciech Zaremba},
  year={2021},
  eprint={2107.03374},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```

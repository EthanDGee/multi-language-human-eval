import pytest
from unittest.mock import patch, mock_open, MagicMock
import json
import os
import sys

# Add the project root directory to the path to import the module
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from human_eval.utils.translator import Translator


class TestTranslator:
    
    @pytest.fixture
    def config_data(self):
        """Test fixture for config data."""
        return {
            "languages": ["spanish", "french"],
            "translation_model": "test-model"
        }
        
    @patch('builtins.open', new_callable=mock_open, read_data='{"languages": ["spanish", "french"], "translation_model": "test-model"}')
    @patch('json.load')
    def test_init(self, mock_json_load, mock_file, config_data):
        """Test Translator initialization."""
        mock_json_load.return_value = config_data
        
        translator = Translator()
        
        assert translator.translation_model == "test-model"
        assert translator.languages == ["spanish", "french"]
        mock_json_load.assert_called_once()
        
    @patch('builtins.open', new_callable=mock_open, read_data='{"languages": ["spanish", "french"], "translation_model": "test-model"}')
    @patch('json.load')
    @patch('human_eval.utils.translator.chat')
    def test_translate_prompt(self, mock_chat, mock_json_load, mock_file, config_data):
        """Test the _translate_prompt method."""
        mock_json_load.return_value = config_data
        mock_chat.return_value = {"message": {"content": "Translated text"}}
        
        translator = Translator()
        result = translator._translate_prompt("spanish", "Test prompt")
        
        assert result == "Translated text"
        mock_chat.assert_called_once()
        
        # Check that the prompt was formatted correctly
        call_args = mock_chat.call_args
        assert call_args[1]['model'] == "test-model"
        assert "Translate the following problem description" in call_args[1]['messages'][0]['content']
        assert "spanish" in call_args[1]['messages'][0]['content']
        assert "Test prompt" in call_args[1]['messages'][0]['content']
        
    @patch('builtins.open', new_callable=mock_open, read_data='{"languages": ["spanish", "french"], "translation_model": "test-model"}')
    @patch('json.load')
    @patch('human_eval.utils.translator.chat')
    @patch('builtins.print')
    def test_translate_problem(self, mock_print, mock_chat, mock_json_load, mock_file, config_data):
        """Test the _translate_problem method."""
        mock_json_load.return_value = config_data
        mock_chat.side_effect = [
            {"message": {"content": "Spanish translation"}},
            {"message": {"content": "French translation"}}
        ]
        
        translator = Translator()
        result = translator._translate_problem("Test problem")
        
        expected = {
            "english": "Test problem",
            "spanish": "Spanish translation",
            "french": "French translation"
        }
        
        assert result == expected
        assert mock_chat.call_count == 2
        
    @patch('builtins.open', new_callable=mock_open, read_data='{"languages": ["spanish"], "translation_model": "test-model"}')
    @patch('json.load')
    @patch('human_eval.utils.translator.chat')
    @patch('builtins.print')
    def test_translate_problem_single_language(self, mock_print, mock_chat, mock_json_load, mock_file):
        """Test _translate_problem with a single language."""
        config_single_lang = {"languages": ["spanish"], "translation_model": "test-model"}
        mock_json_load.return_value = config_single_lang
        mock_chat.return_value = {"message": {"content": "Spanish translation"}}
        
        translator = Translator()
        result = translator._translate_problem("Test problem")
        
        expected = {
            "english": "Test problem",
            "spanish": "Spanish translation"
        }
        
        assert result == expected
        mock_chat.assert_called_once()
        
    @patch('builtins.open', new_callable=mock_open, read_data='{"languages": [], "translation_model": "test-model"}')
    @patch('json.load')
    @patch('human_eval.utils.translator.chat')
    @patch('builtins.print')
    def test_translate_problem_no_languages(self, mock_print, mock_chat, mock_json_load, mock_file):
        """Test _translate_problem with no target languages."""
        config_no_lang = {"languages": [], "translation_model": "test-model"}
        mock_json_load.return_value = config_no_lang
        
        translator = Translator()
        result = translator._translate_problem("Test problem")
        
        expected = {"english": "Test problem"}
        
        assert result == expected
        mock_chat.assert_not_called()
        
    @patch('builtins.open', new_callable=mock_open, read_data='{"languages": ["spanish"], "translation_model": "test-model"}')
    @patch('json.load')
    @patch('human_eval.utils.translator.read_problems')
    @patch('human_eval.utils.translator.write_jsonl')
    @patch('human_eval.utils.translator.chat')
    @patch('builtins.print')
    def test_translate_dataset(self, mock_print, mock_chat, mock_write_jsonl, mock_read_problems, mock_json_load, mock_file):
        """Test the translate_dataset method."""
        mock_json_load.return_value = {"languages": ["spanish"], "translation_model": "test-model"}
        
        # Mock problems data
        mock_problems = {
            "test_task_1": {
                "task_id": "test_task_1",
                "prompt": "Test prompt 1",
                "other_field": "value1"
            },
            "test_task_2": {
                "task_id": "test_task_2", 
                "prompt": "Test prompt 2",
                "other_field": "value2"
            }
        }
        mock_read_problems.return_value = mock_problems
        
        mock_chat.return_value = {"message": {"content": "Spanish translation"}}
        
        translator = Translator()
        translator.translate_dataset()
        
        # Verify read_problems was called
        mock_read_problems.assert_called_once()
        
        # Verify write_jsonl was called with correct filename
        mock_write_jsonl.assert_called_once()
        args, kwargs = mock_write_jsonl.call_args
        assert args[0] == "data/TranslatedHumanEval.jsonl.gz"
        
        # Check the structure of translated data
        translated_data = args[1]
        assert len(translated_data) == 2
        
        # Check first problem
        first_problem = translated_data[0]
        assert first_problem["task_id"] == "test_task_1"
        assert "prompts" in first_problem
        assert "english" in first_problem["prompts"]
        assert "spanish" in first_problem["prompts"]
        assert "prompt" not in first_problem  # Original prompt should be removed
        assert first_problem["other_field"] == "value1"  # Other fields preserved
        
        # Check second problem
        second_problem = translated_data[1]
        assert second_problem["task_id"] == "test_task_2"
        assert "prompts" in second_problem
        assert second_problem["other_field"] == "value2"


class TestTranslatorIntegration:
    """Integration tests that test the overall functionality."""
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    @patch('human_eval.utils.translator.read_problems')
    @patch('human_eval.utils.translator.write_jsonl')
    @patch('human_eval.utils.translator.chat')
    @patch('builtins.print')
    def test_full_translation_workflow(self, mock_print, mock_chat, mock_write_jsonl, mock_read_problems, mock_json_load, mock_file):
        """Test the complete translation workflow."""
        # Setup config
        config_data = {
            "languages": ["spanish", "french"],
            "translation_model": "test-model"
        }
        mock_json_load.return_value = config_data
        
        # Setup problems
        mock_problems = {
            "task_1": {
                "task_id": "task_1",
                "prompt": "Write a function that adds two numbers.",
                "description": "Simple addition function"
            }
        }
        mock_read_problems.return_value = mock_problems
        
        # Setup chat responses
        mock_chat.side_effect = [
            {"message": {"content": "Escribe una función que suma dos números."}},
            {"message": {"content": "Écrivez une fonction qui additionne deux nombres."}}
        ]
        
        translator = Translator()
        translator.translate_dataset()
        
        # Verify the workflow
        mock_read_problems.assert_called_once()
        assert mock_chat.call_count == 2  # One for each language
        mock_write_jsonl.assert_called_once()
        
        # Check the final data structure
        args, _ = mock_write_jsonl.call_args
        translated_data = args[1]
        
        assert len(translated_data) == 1
        result = translated_data[0]
        
        assert result["task_id"] == "task_1"
        assert result["description"] == "Simple addition function"  # Preserved
        assert "prompts" in result
        assert result["prompts"]["english"] == "Write a function that adds two numbers."
        assert result["prompts"]["spanish"] == "Escribe una función que suma dos números."
        assert result["prompts"]["french"] == "Écrivez une fonction qui additionne deux nombres."


class TestTranslatorEdgeCases:
    """Test edge cases and error conditions."""
    
    @patch('builtins.open', new_callable=mock_open, read_data='{"languages": ["spanish"], "translation_model": "test-model"}')
    @patch('json.load')
    @patch('human_eval.utils.translator.chat')
    def test_translate_prompt_empty_response(self, mock_chat, mock_json_load, mock_file):
        """Test handling of empty translation response."""
        mock_json_load.return_value = {"languages": ["spanish"], "translation_model": "test-model"}
        mock_chat.return_value = {"message": {"content": ""}}
        
        translator = Translator()
        result = translator._translate_prompt("spanish", "Test prompt")
        
        assert result == ""
        
    @patch('builtins.open', new_callable=mock_open, read_data='{"languages": ["spanish"], "translation_model": "test-model"}')
    @patch('json.load')
    @patch('human_eval.utils.translator.chat')
    def test_translate_prompt_with_special_characters(self, mock_chat, mock_json_load, mock_file):
        """Test translation with special characters in prompt."""
        mock_json_load.return_value = {"languages": ["spanish"], "translation_model": "test-model"}
        mock_chat.return_value = {"message": {"content": "Traducción con caracteres especiales: ñáéíóú"}}
        
        translator = Translator()
        result = translator._translate_prompt("spanish", "Function with special chars: @#$%^&*()")
        
        assert result == "Traducción con caracteres especiales: ñáéíóú"
        
    @patch('builtins.open', side_effect=FileNotFoundError("Config file not found"))
    def test_init_missing_config_file(self, mock_file):
        """Test handling of missing config file."""
        with pytest.raises(FileNotFoundError):
            Translator()
            
    @patch('builtins.open', new_callable=mock_open, read_data='{"invalid": "json"}')
    @patch('json.load')
    def test_init_invalid_config(self, mock_json_load, mock_file):
        """Test handling of invalid config structure."""
        mock_json_load.return_value = {"invalid": "json"}
        
        with pytest.raises(KeyError):
            Translator()
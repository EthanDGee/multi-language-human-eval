import json


class Translator:
    def __init__(self):
        config = json.load(open("config.json", "r"))

        self.translation_model = config["translation_model"]
        self.languages = config["languages"]

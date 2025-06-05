import json
import re

class Tokenizer:
    def __init__(self, length_sentence):

        self.length_sentence = length_sentence

        self.PAD_TOKEN = "<pad>"
        self.UNK_TOKEN = "<unk>"
        self.SEP_TOKEN = "<sep>"
        self.ANS_TOKEN = "<ans>"

        self.tokens = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
            self.SEP_TOKEN: 2,
            self.ANS_TOKEN: 3,
        }

    def tokenize_text(self, text):
        text = text.lower()
        words = re.findall(r"<[^<>]+>|[\w]+", text)
        tokens = []

        for word in words:
            if word not in self.tokens:
                self.tokens[word] = len(self.tokens)

            tokens.append(self.tokens[word])

        return tokens

    def format_tokens(self, input):
        while(len(input) < self.length_sentence):
            input.append(self.tokens[self.PAD_TOKEN])

        if (len(input) > self.length_sentence):
            input = input[:self.length_sentence]

        return input
    
    def tokenize_and_format(self, input, output):
        input = self.tokenize_text(input)
        output = self.tokenize_text(output)

        input = self.format_tokens(input)
        output = self.format_tokens(output)

        return input, output

    def open_file(self, name_file):
        with open(name_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data

    def save_tokens(self, name_file):
        name = name_file + ".jsonl"

        with open(name, "w", encoding="utf-8") as f:
            for word, idx in self.tokens.items():
                json.dump({"word": word, "id": idx}, f, ensure_ascii=False)
                f.write("\n")

        print(f"'{name}' has been created")

    def save_sentences(self, sentences, name_file):
        name = name_file + ".jsonl"

        with open(name, "w", encoding="utf-8") as f:
            for item in sentences:
                json.dump(item, f)
                f.write("\n")

        print(f"'{name}' has been created")

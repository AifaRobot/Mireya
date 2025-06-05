import torch
import os
from transformers.transformer import Transformer
from .components import DecoderLayer, Embedding, EncoderLayer, Linear, PositionalEncoding
from .components.utils import BatchLoader, draw_plot

"""The Translator class is designed to train and use a Transformer-based translation model. This model can translate
sentences from a source language (e.g. Spanish) to a target language (e.g. English)."""
class TranslatorTransformer(Transformer):

    def __init__(
        self,
        source_vocabulary,
        target_vocabulary,
        embedding_dimension,
        number_heads,
        feed_forward_dimension,
        limit_sequence_length,
        dropout,
        learning_rate = 0.0001,
        device = 'cpu',
        batch_size = 32,
        batch_num_workers = 0,
        batch_shuffle = True,
        save_path = '/saved'
    ):

        self.device = device
        self.source_vocabulary = source_vocabulary
        self.target_vocabulary = target_vocabulary
        self.limit_sequence_length = limit_sequence_length
        self.batch_size = batch_size
        self.batch_num_workers = batch_num_workers
        self.batch_shuffle = batch_shuffle
        self.save_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + save_path

        self.encoders = self.create_encoder_stack([
            Embedding(len(source_vocabulary), embedding_dimension).to(self.device),
            PositionalEncoding(embedding_dimension, limit_sequence_length).to(self.device),
            EncoderLayer(embedding_dimension, number_heads, feed_forward_dimension, dropout).to(self.device),
            EncoderLayer(embedding_dimension, number_heads, feed_forward_dimension, dropout).to(self.device),
            EncoderLayer(embedding_dimension, number_heads, feed_forward_dimension, dropout).to(self.device),
            EncoderLayer(embedding_dimension, number_heads, feed_forward_dimension, dropout).to(self.device),
            EncoderLayer(embedding_dimension, number_heads, feed_forward_dimension, dropout).to(self.device),
            EncoderLayer(embedding_dimension, number_heads, feed_forward_dimension, dropout).to(self.device),
            EncoderLayer(embedding_dimension, number_heads, feed_forward_dimension, dropout).to(self.device),
        ])

        self.decoders = self.create_decoder_stack([
            Embedding(len(target_vocabulary), embedding_dimension).to(self.device),
            PositionalEncoding(embedding_dimension, limit_sequence_length).to(self.device),
            DecoderLayer(embedding_dimension, number_heads, feed_forward_dimension, dropout).to(self.device),
            DecoderLayer(embedding_dimension, number_heads, feed_forward_dimension, dropout).to(self.device),
            DecoderLayer(embedding_dimension, number_heads, feed_forward_dimension, dropout).to(self.device),
            DecoderLayer(embedding_dimension, number_heads, feed_forward_dimension, dropout).to(self.device),
            DecoderLayer(embedding_dimension, number_heads, feed_forward_dimension, dropout).to(self.device),
            DecoderLayer(embedding_dimension, number_heads, feed_forward_dimension, dropout).to(self.device),
            Linear(embedding_dimension, len(target_vocabulary)).to(self.device),
        ])

        self.encoders.load(self.save_path)
        self.decoders.load(self.save_path)

        self.criterion = torch.nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(self.encoders.parameters() + self.decoders.parameters(), lr=learning_rate)

    def train(self, path_file, num_epochs):
        """The transformer enters evaluation mode causing Dropout to be activated, allowing the full predictive power
        of the model to be used"""
        self.encoders.train()
        self.decoders.train()

        batch_loader = BatchLoader(
            path_file = path_file, 
            batch_size = self.batch_size, 
            num_workers = self.batch_num_workers, 
            shuffle = self.batch_shuffle, 
            device = self.device
        )

        losses = []

        for epoch in range(num_epochs):
            input, target = batch_loader.get_batch()

            self.optimizer.zero_grad()

            encoder_out, encoder_mask = self.encoders(encoder=input)
            decoder_out, _ = self.decoders(
                decoder=target[:, :-1],
                encoder=encoder_out,
                encoder_mask=encoder_mask,
            )

            loss = self.criterion(
                decoder_out.view(-1, decoder_out.size(-1)),
                target[:, 1:].contiguous().view(-1),
            )
            loss.backward()

            self.optimizer.step()

            losses.append(loss.item())

            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item() / len(input)}")

        self.encoders.save(self.save_path)
        self.decoders.save(self.save_path)
        
        draw_plot(
            history = losses,
            xlabel = "Epochs",
            ylabel = "Loss", 
            save_path = self.save_path, 
        )

    def forward(self, sentence):
        
        """The transformer enters evaluation mode causing Dropout to turn off, allowing the full predictive power
        of the model to be used"""
        self.encoders.eval()
        self.decoders.eval()
        
        """An arrangement is created with the words of the sentence in input vocabulary (Spanish)"""
        words = sentence.split()
        
        """An array of tokens is generated, where each token is an integer that represents one of the words in the sentence"""
        input_tokens = torch.tensor([self.source_vocabulary[word] for word in words], dtype=torch.long).unsqueeze(0).to(self.device)
        
        """An array is created that will store all the tokens that will represent the translation. The array begins with a first
        token <sos> which means Start of Sentence"""
        generated_tokens = [self.target_vocabulary["<sos>"]]

        for _ in range(self.limit_sequence_length):
            generated_tokens_tensor = torch.tensor(generated_tokens, dtype=torch.long).unsqueeze(0).to(self.device)

            """The tokenized sentence to be translated (input_token) and the tokenized sentence generated so far are used,
            and returns an array where each element is the probability of choosing one of the tokens (words) of the target
            vocabulary (English)"""
            encoder_out, encoder_mask = self.encoders(encoder=input_tokens)
            decoder_out, _ = self.decoders(
                decoder=generated_tokens_tensor,
                encoder=encoder_out,
                encoder_mask=encoder_mask,
            )

            """The token (word) with the highest probability is chosen to be added to the translated sentence that is being
            formed and that is tokenized"""
            next_token = decoder_out.argmax(-1)[:, -1].item()

            generated_tokens.append(next_token)

            """If the next token is <pad> then stop generating tokens"""
            if next_token == self.target_vocabulary["<pad>"]:
                break

        """An array is created where each element is a word that is equivalent to each of the tokens found in the
        generated_tokens array"""
        translation = [
            list(self.target_vocabulary.keys())[list(self.target_vocabulary.values()).index(token)]
            for token in generated_tokens
        ]

        print("Translate:", sentence + " = " + " ".join(translation))

    def test(self, path_file):
        """The transformer enters evaluation mode causing Dropout to be activated, allowing the full predictive power
        of the model to be used"""
        self.decoders.eval()

        batch_loader = BatchLoader(
            path_file = path_file,
            batch_size = self.batch_size,
            num_workers = self.batch_num_workers,
            shuffle = self.batch_shuffle,
            device = self.device
        )

        inputs, targets = batch_loader.get_all()

        total_ok_tests = 0

        for input, target in zip(inputs, targets):
            message = ''
            generated_tokens = []

            for token in input:
                word = list(self.source_vocabulary.keys())[list(self.source_vocabulary.values()).index(token)]
                
                if (word == '<pad>'):
                    break

                message += word + ' '  
                generated_tokens.append(token)

            message += ' <---(*TRANSLATE*)---> '
            generated_tokens = [self.target_vocabulary["<sos>"]]
            input = input.to(self.device).unsqueeze(0)

            for _ in range(self.limit_sequence_length):
                generated_tokens_tensor = torch.tensor(generated_tokens, dtype=torch.long).unsqueeze(0).to(self.device)

                """The tokenized sentence to be translated (input_token) and the tokenized sentence generated so far are used,
                and returns an array where each element is the probability of choosing one of the tokens (words) of the target
                vocabulary (English)"""
                encoder_out, encoder_mask = self.encoders(encoder=input)

                decoder_out, _ = self.decoders(
                    decoder=generated_tokens_tensor,
                    encoder=encoder_out,
                    encoder_mask=encoder_mask,
                )

                """The token (word) with the highest probability is chosen to be added to the translated sentence that is being
                formed and that is tokenized"""
                next_token = decoder_out.argmax(-1)[:, -1].item()

                generated_tokens.append(next_token)

                """If the next token is <pad> then stop generating tokens"""
                if next_token == self.target_vocabulary["<pad>"]:
                    break

                message += ' ' + list(self.target_vocabulary.keys())[list(self.target_vocabulary.values()).index(next_token)]

            result = '\033[91mNO OK\033[0m'

            if (torch.equal(generated_tokens_tensor.squeeze(0).to(self.device), target[target != 0].to(self.device))):
                total_ok_tests += 1
                result = '\033[92mOK\033[0m'

            print(message + ' //// STATUS: ' + result)
        
        print('Success Percentage: ' + str((total_ok_tests / len(inputs)) * 100) + ' %')
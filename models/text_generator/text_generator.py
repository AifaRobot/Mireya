import torch
from transformer.transformer import Transformer

'''The Translator class is designed to train and use a Transformer-based translation model. This model can translate 
sentences from a source language (e.g. Spanish) to a target language (e.g. English).'''
class TextGenerator:
    def __init__(self, source_vocabulary, embedding_dimension, number_heads, number_layers, 
    feed_forward_dimension, limit_sequence_length, dropout, learning_rate = 0.0001, use_encoder_layers = True):
        self.source_vocabulary = source_vocabulary
        self.embedding_dimension = embedding_dimension
        self.number_heads = number_heads
        self.number_layers = number_layers
        self.feed_forward_dimension = feed_forward_dimension
        self.limit_sequence_length = limit_sequence_length
        self.dropout = dropout
        self.use_encoder_layers = use_encoder_layers

        self.transformer = Transformer(
            len(source_vocabulary), 
            len(source_vocabulary), 
            self.embedding_dimension, 
            self.number_heads, 
            self.number_layers, 
            self.feed_forward_dimension, 
            self.limit_sequence_length, 
            self.dropout,
            self.use_encoder_layers
        )

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

        self.optimizer = torch.optim.Adam(self.transformer.parameters(), lr=learning_rate)

    def train(self, sentences, num_epochs):
        '''The transformer enters evaluation mode causing Dropout to be activated, allowing the full predictive power 
        of the model to be used'''
        self.transformer.train()

        input_batch = torch.tensor([sentence['input'][0] for sentence in sentences])
        target_batch = torch.tensor([sentence['output'][0] for sentence in sentences])

        for epoch in range(num_epochs):
            self.optimizer.zero_grad()

            output = self.transformer(input_batch)

            loss = self.criterion(output.view(-1, output.size(-1)), target_batch.view(-1))
            loss.backward()

            self.optimizer.step()

            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item() / len(sentences)}")

    def talk(self, sentence):
        '''The transformer enters evaluation mode causing Dropout to turn off, allowing the full predictive power of 
        the model to be used'''
        self.transformer.eval()

        '''An arrangement is created with the words of the sentence in input vocabulary (Spanish)'''
        words = sentence.split()

        '''An array of tokens is generated, where each token is an integer that represents one of the words in the sentence.
        An array is created that will store all the tokens that will represent the translation. The array begins with a first 
        token <SOS> which means Start of Sentence'''
        generated_tokens = [self.source_vocabulary[word] for word in words]
        generated_tokens.insert(0, self.source_vocabulary['<SOS>'])

        for _ in range(self.limit_sequence_length):
            generated_tokens_tensor = torch.tensor(generated_tokens, dtype=torch.long).unsqueeze(0)

            '''We deliver the generated tokens to the transformer to obtain the next word that fits the content.'''
            output = self.transformer(generated_tokens_tensor)
            
            '''The token (word) with the highest probability is chosen to be added to the translated sentence that is being 
            formed and that is tokenized'''
            next_token = output.argmax(-1)[:, -1].item()
            
            generated_tokens.append(next_token)
            
            '''If the next token is <EOS> then stop generating tokens. <EOS> means End of Sentence'''
            if next_token == self.source_vocabulary['<EOS>']:
                break

        '''An array is created where each element is a word that is equivalent to each of the tokens found in the generated_
        tokens array'''
        translation = [list(self.source_vocabulary.keys())[list(self.source_vocabulary.values()).index(token)] for token in generated_tokens]
        
        print("Talk:", sentence + " = " + " ".join(translation))

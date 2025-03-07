import torch
import torch.nn as nn
from transformer.utils.positional_encoding import PositionalEncoding
from transformer.utils.encoder_layer import EncoderLayer
from transformer.utils.decoder_layer import DecoderLayer

'''A transformer is a type of neural network architecture originally designed for natural language processing (NLP) tasks, 
but which has proven to be very versatile and is now applied in various fields, such as computer vision and bioinformatics. 
It was introduced in the article "Attention Is All You Need" by Vaswani et al. in 2017.

Main characteristics of transformers:
        
1. Attention:

    The key component is the Self-Attention mechanism, which allows the network to focus on different parts of an input sequence 
    while processing each token (word, symbol, etc.).
        
    For example, in a sentence, the transformer can assign greater weight to the most relevant words to understand the context.
    
2. Layered structure:

    Transformers are composed of multiple layers of encoders (encoder) and decoders (decoder).
            
        The encoder takes the input and generates rich internal representations (embeddings).
            
        The decoder uses those representations to generate an output (such as a translation or generated text).
        
    In some cases, such as VER, only the encoder is used; in others, such as GPT, only the decoder is used.'''
class Transformer(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, embedding_dimension, number_heads, number_layers, 
        feed_forward_dimension, max_sequence_length, dropout, use_encoder_layers):
        
        super(Transformer, self).__init__()

        self.use_encoder_layers = use_encoder_layers
        
        '''
            Text is converted into numbers (indexes) through a process called tokenization. However, these indices do not 
            contain semantic information, so we need to map them to dense vectors that capture relationships between the 
            tokens. This is where nn.Embedding comes into play.

            For example, if you have a vocabulary of 5 words ([0, 1, 2, 3, 4]) and each embedding has dimension 3, the embeddings table could be something like this:

            Vocabulary: [0, 1, 2, 3, 4]
            Embeddings:
                0 -> [0.1, 0.2, 0.3]
                1 -> [0.5, 0.6, 0.7]
                2 -> [0.9, 1.0, 1.1]
                3 -> [1.3, 1.4, 1.5]
                4 -> [1.7, 1.8, 1.9]

            If the input is [2, 3], the output will be:

            [[0.9, 1.0, 1.1],
            [1.3, 1.4, 1.5]]
        '''
        self.decoder_embedding = nn.Embedding(output_vocab_size, embedding_dimension)
        self.decoder_layers = nn.ModuleList([DecoderLayer(embedding_dimension, number_heads, feed_forward_dimension, dropout, use_encoder_layers) for _ in range(number_layers)])

        '''The Encoder and Decoder layers are created. In the original Paper "Attention Is All You Need" there were 6 
        Encoder layers and 6 Decoder layers'''
        if (self.use_encoder_layers):
            self.encoder_layers = nn.ModuleList([EncoderLayer(embedding_dimension, number_heads, feed_forward_dimension, dropout) for _ in range(number_layers)])
            self.encoder_embedding = nn.Embedding(input_vocab_size, embedding_dimension)

        self.linear_layer_output = nn.Linear(embedding_dimension, output_vocab_size)

        '''When calculating embeddings for tokens (encoder_embedding and decoder_embedding), Dropout is applied after
        combine them with positional_encoding to ensure that the model does not depend too much on
        specific patterns in token representations'''
        self.dropout = nn.Dropout(dropout)

        '''This layer will embed the word representations with positional information'''
        self.positional_encoding = PositionalEncoding(embedding_dimension, max_sequence_length)

    def generate_causal_mask(self, encoder_input):
        '''Compares each element of input with 0. Example: If input = [[1, 2, 3, 0, 0]], then: [[True, True, True, False, False]]'''
        encoder_input_mask = (encoder_input != 0).unsqueeze(1).unsqueeze(3)

        sequence_length = encoder_input.size(1)

        '''
            Creates a tensor filled with ones of form (1, sequence_length, sequence_length). This tensor will be the basis for 
            building the causal mask.

            Extracts the top part of the triangular matrix, excluding the main diagonal. For sequence_length = 4, the result will be:
 
            [[0, 1, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 0]]

            The values are inverted (0 → 1, 1 → 0). This creates a lower triangular mask:

            [[1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 1]]

            Converts numeric values to boolean (1 → True, 0 → False):

            [[True, False, False, False],
            [True,  True, False, False],
            [True,  True,  True, False],
            [True,  True,  True,  True]]

            The purpose of the nopeal_mask is to ensure that a token at position i can only "see" previous tokens (or itself) and 
            not future ones.
        '''
        nopeak_mask = (1 - torch.triu(torch.ones(1, sequence_length, sequence_length), diagonal=1)).bool()

        '''
            Combine the fill mask (output_mask) and the causal mask (nopeak_mask) using a logical "AND" operation.

            This ensures that padding (0) tokens are ignored and the causality constraint (don't look into the future) is respected.

            Example: If output_mask = [[[[True], [True], [True], [False]]]] and nopeak_mask is:

            [[True, False, False, False],
            [True,  True, False, False],
            [True,  True,  True, False],
            [True,  True,  True,  True]]

            The combined result (output_mask) will be:

            [[[[True, False, False, False],
            [True,  True, False, False],
            [True,  True,  True, False],
            [False, False, False, False]]]]
        '''
        encoder_input_mask = encoder_input_mask & nopeak_mask

        return encoder_input_mask

    def forward(self, decoder_input, encoder_input = None):
        encoder_output, encoder_output_mask = None, None

        '''It starts by feeding the self.encoder_layers stack with the source_embedded so that the encoder_output is 
        then passed from one layer to the next until the stack is complete'''
        if (self.use_encoder_layers):
            encoder_output_mask = (encoder_input != 0).unsqueeze(1).unsqueeze(2)

            '''It generates embedding the entries with positional information'''
            encoder_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(encoder_input)))
            
            '''Start by feeding the self.decoder_layers stack with the target_embedded so that the encoder_output is then 
            passed from one layer to the next until the stack is complete'''
            encoder_output = encoder_embedded
            for encoder_layer in self.encoder_layers:
                encoder_output = encoder_layer(encoder_output, encoder_output_mask)

        '''The generate_causal_mask function generates a "causal mask". Causal masks in transformers are a mechanism used to 
        control attention in self-regressive attention layers, such as those found in generative language models. These 
        masks ensure that each token can only "see" or attend to tokens before it in the sequence, and not future tokens'''
        decoder_output_mask = self.generate_causal_mask(decoder_input)

        '''It generates embedding the entries with positional information'''
        decoder_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(decoder_input)))

        '''Start by feeding the self.decoder_layers stack with the target_embedded so that the encoder_output is then 
        passed from one layer to the next until the stack is complete'''
        decoder_output = decoder_embedded
        for decoder_layer in self.decoder_layers:
            decoder_output = decoder_layer(decoder_output, decoder_output_mask, encoder_output, encoder_output_mask)

        '''It is the final step to map the internal representations of the model (decoder_output) to an interpretable 
        space: a probability distribution over the target vocabulary'''
        output = self.linear_layer_output(decoder_output)

        return output
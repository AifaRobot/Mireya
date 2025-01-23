import torch.nn as nn
from transformer.utils.multi_head_attention import MultiHeadAttention
from transformer.utils.feed_forward import FeedForward

'''
    The EncoderLayer class implements one of the encoder layers in the Transformer model. Its main function is to process 
    input representations and generate richer representations by capturing relationships between tokens in the input sequence.
'''
class EncoderLayer(nn.Module):
    def __init__(self, embedding_dimension, number_heads, feed_forward_dimension, dropout):
        super(EncoderLayer, self).__init__()

        # Self-attention allows an element in a sequence (such as a word or token) to pay attention to other elements 
        # in the same sequence. This is crucial to capture dependencies within the same entry
        self.self_attention = MultiHeadAttention(embedding_dimension, number_heads)
        
        # It is a feed-forward network that is independently applied to each position in the sequence to learn complex 
        # nonlinear transformations to enrich representations and complement the capacity of attention
        self.feed_forward = FeedForward(embedding_dimension, feed_forward_dimension)
        
        # Normalization layers stabilize the learning process of the Decoder layers
        self.normalizer_layer_1 = nn.LayerNorm(embedding_dimension)
        self.normalizer_layer_2 = nn.LayerNorm(embedding_dimension)
        
        # Dropout introduces noise during training, forcing the network to learn more robust and generalizable representations 
        # instead of overly relying on specific combinations of neurons
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attention_output = self.self_attention(x, x, x, mask)
        x = self.normalizer_layer_1(x + self.dropout(attention_output))
        ff_output = self.feed_forward(x)
        x = self.normalizer_layer_2(x + self.dropout(ff_output))
        
        return x
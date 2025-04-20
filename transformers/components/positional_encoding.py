import math
import torch

"""The PositionalEncoding class adds information about the position of each token in a sequence, allowing the Transformer model
to be aware of the sequential structure of the data. This is crucial because, unlike recurrent networks (RNNs) or convolutional
networks (CNNs), Transformers do not process data in an implicit sequential order.

Why is PositionalEncoding necessary?

Transformers process a sequence as a set of tokens in parallel. However, the order of the tokens is crucial to understanding
the meaning of the sequence. For example:

In "The Cat Chases the Mouse", the order matters to know who is chasing who.

PositionalEncoding is responsible for injecting information about the position of each token in the sequence, allowing the
model to differentiate between the first, second, third token, etc.

Properties of sine and cosine in linear combinations

The sine and cosine functions have a key property that makes learning positional relationships in Transformers easier:

sin(a+b)=sin(a)cos(b)+cos(a)sin(b)
cos(a+b)=cos(a)cos(b)-sin(a)sin(b)

This means that by combining positions through linear calculations, the model can deduce complex positional relationships.

For example for two positions pos 1 and pos 2, the differences in their sine and cosine values can be used by the model to
infer how far apart they are in the sequence. This is especially useful for tasks where relative relationships are important,
such as machine translation or temporal sequence analysis."""
class PositionalEncoding(torch.nn.Module):

    def __init__(self, embedding_dimension, max_sequence_length):
        super(PositionalEncoding, self).__init__()
        
        """Create a tensor to store the position vectors"""
        pe = torch.zeros(max_sequence_length, embedding_dimension)
        
        """Create a tensor with the sequence positions (0, 1, 2, ..., max_sequence_length - 1)"""
        position = torch.arange(0, max_sequence_length, dtype=torch.float).unsqueeze(1)
        
        """Calculate frequencies (div_term) based on model dimension"""
        div_term = torch.exp(
            torch.arange(0, embedding_dimension, 2).float() *
            -(math.log(10000.0) / embedding_dimension))
        
        """Apply sine and cosine for even and odd indices"""
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        """Add a batch dimension to facilitate calculation"""
        self.register_buffer("pe", pe.unsqueeze(0))

        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x, _):
        
        """Add the positional encoding to the token embedding"""
        return self.dropout(x + self.pe[:, :x.size(1)])

"""The PatchPositionalEncoding is commonly used in Vision Transformer (ViT) type models, where an
image is divided into patches (smaller blocks) and each is represented with an embedding."""
class PatchPositionalEncoding(torch.nn.Module):

    def __init__(self, embedding_dimension, image_size, patch_size):
        super(PatchPositionalEncoding, self).__init__()
        self.num_patches = (image_size // patch_size)**2
        
        """Learnable parameters (cls_token and pos_embed) are used here for one key reason: they allow the model to learn useful
        information during training instead of using fixed values"""
        
        """Special token added at the beginning of the sequence. In classification, it is used to represent the entire image"""
        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, embedding_dimension))
        
        """positional embeddings that are added to each patch (plus one for the cls_token).
        pos_embed is used because the transformer must learn the spatial relationship between patches"""
        self.pos_embed = torch.nn.Parameter(
            torch.randn(1, self.num_patches + 1, embedding_dimension))

    def forward(self, _, memory):
        B = (memory.get_original_encoder_tensor().shape[0]
             if memory.get_stack() == "encoder" else memory.get_original_decoder_tensor().shape[0])
        x = (memory.get_encoder_tensor()
             if memory.get_stack() == "encoder" else memory.get_decoder_tensor())
        
        """It takes the [CLS] token (which is an nn.Parameter of the form [1, 1, embedding_dim]) and duplicates it B times, one for
        each element in the batch. Result: cls_tokens has the form [B, 1, embedding_dim]."""
        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        """Concatenates the cls_token to the start of the patch sequence. Now x is in the form [B, num_patches + 1, embedding_dim]."""
        x = torch.cat((cls_tokens, x), dim=1)
        
        """Adds the learnable positional embeddings (self.pos_embed) to each vector in the sequence, to give a notion of position."""
        x = x + self.pos_embed

        return x
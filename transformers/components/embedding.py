import torch.nn as nn

"""An embedding is a way of representing discrete data (such as words or tokens) as continuous
vectors of numbers. Instead of using words like "hello" or "world," the model works with vectors like:

[0.23, -1.04, 0.88, ..., 0.15] # fixed size, e.g.: 512 dimensions

Why do Transformers need this?

Transformers don't understand raw text or token IDs; they need vectors that have semantic and numerical structure.

So embedding is used to transform:

"hello" → ID 384 → embedding vector x384 = [0.5, -0.3, ...]

This vector is what the Transformer actually processes."""
class Embedding(nn.Module):

    def __init__(self, len_source_vocabulary, embedding_dimension):
        super(Embedding, self).__init__()

        self.embedding = nn.Embedding(len_source_vocabulary, embedding_dimension)

    def get_parameters(self):
        return self.embedding.parameters()

    def forward(self, x, _):
        return self.embedding(x)

"""Patch embedding is "embedding for images," and it's incredibly important in models like the Vision
Transformer (ViT). While regular Transformers use word embedding, the ViT uses image patches as if they
were "words."

In the Vision Transformer, instead of processing text, we process images. But Transformers don't understand
images directly, so:

Let's assume an image of size 224x224x3 (RGB):

We divide the image into patches, for example, 16x16.
→ That gives 14x14 = 196 patches.

We flatten each patch (16x16x3 = 768 values per patch).

We project each patch with a nn.Linear(768, D) layer to a D-dimensional vector (embedding_dim, for example, 512).

Now we have 196 vectors, each of size 512. These are your "tokens" for the Transformer."""
class PatchEmbedding(nn.Module):

    def __init__(self, image_size, patch_size, input_channels, embedding_dimension):
        super().__init__()

        self.num_patches = (image_size // patch_size)**2
        self.proj = nn.Conv2d(
            input_channels,
            embedding_dimension,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x, _):
        
        """Applies conv2d and converts the image into a grid of embedded patches."""
        x = self.proj(x) # [B, embedding_dimension, num_patches^0.5, num_patches^0.5]
        
        """Flattens the spatial dimensions (H', W') into one to have num_patches"""
        x = x.flatten(2) # [B, embedding_dimension, num_patches]
        
        """Reorder so that the final result is like a sequence of patches as if they were tokens,
        each with its embedding vector"""
        x = x.transpose(1, 2) # [B, num_patches, embedding_dimension]}

        return x
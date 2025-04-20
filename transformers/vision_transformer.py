import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers.transformer import Transformer
from .components import EncoderLayer, Linear, LinearNormalization, PatchEmbedding, PatchPositionalEncoding, ShapeTransformation

"""The Translator class is designed to train and use a Transformer-based translation model. This model can translate
sentences from a source language (e.g. Spanish) to a target language (e.g. English)."""
class VisionTransformer(Transformer):

    def __init__(
        self,
        embedding_dimension,
        number_heads,
        feed_forward_dimension,
        dropout,
        image_size,
        input_channels,
        patch_size,
        num_classes,
        learning_rate=0.0001,
    ):

        self.encoders = self.create_encoder_stack([
            PatchEmbedding(
                image_size=image_size,
                input_channels=input_channels,
                patch_size=patch_size,
                embedding_dimension=embedding_dimension,
            ),
            PatchPositionalEncoding(embedding_dimension, image_size=image_size, patch_size=patch_size),
            EncoderLayer(embedding_dimension, number_heads, feed_forward_dimension, dropout),
            EncoderLayer(embedding_dimension, number_heads, feed_forward_dimension, dropout),
            EncoderLayer(embedding_dimension, number_heads, feed_forward_dimension, dropout),
            EncoderLayer(embedding_dimension, number_heads, feed_forward_dimension, dropout),
            EncoderLayer(embedding_dimension, number_heads, feed_forward_dimension, dropout),
            EncoderLayer(embedding_dimension, number_heads, feed_forward_dimension, dropout),
            ShapeTransformation(lambda x: x[:, 0]),
            LinearNormalization(embedding_dimension),
            Linear(embedding_dimension, num_classes),
        ])

        self.criterion = torch.nn.CrossEntropyLoss()

        self.optimizer = torch.optim.AdamW(self.encoders.parameters(), lr=learning_rate)

    def train(self, train_loader, num_epochs):
        
        """The transformer enters evaluation mode causing Dropout to be activated, allowing the full predictive power
        of the model to be used"""
        self.encoders.train()

        for epoch in range(num_epochs):
            losses = []

            for images, labels in train_loader:

                self.optimizer.zero_grad()

                encoder_output, _ = self.encoders(encoder=images, generate_encoder_mask=False)

                loss = self.criterion(encoder_output, labels)
                loss.backward()
                self.optimizer.step()

                losses.append(loss.item())

            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {np.array(losses).mean()}")

    def forward(self, train_loader):

        for _ in range(50):
            train_dataset = train_loader

            random_idx = random.randint(0, len(train_dataset) - 1)

            image, label = train_dataset[random_idx]

            image = image.unsqueeze(0)

            encoder_output, _ = self.encoders(encoder=image, generate_encoder_mask=False)

            print(f"Transformer Prediction {encoder_output.squeeze(0).argmax()}, Label: {label}, Error: {(label == encoder_output.squeeze(0).argmax()) == False}")

            plt.imshow(image.squeeze(), cmap="gray")
            plt.title(f"Label: {label}")
            plt.axis("off")
            plt.show()
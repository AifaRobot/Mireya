import random
import matplotlib.pyplot as plt
import torch
import os
from transformers.transformer import Transformer
from .components import EncoderLayer, Linear, LinearNormalization, PatchEmbedding, PatchPositionalEncoding, ShapeTransformation
from .components.utils import BatchLoader, draw_plot

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
        learning_rate = 0.0001,
        device = 'cpu',
        batch_size = 32,
        batch_num_workers = 0,
        batch_shuffle = True,
        save_path = '/saved'
    ):

        self.device = device
        self.batch_size = batch_size
        self.batch_num_workers = batch_num_workers
        self.batch_shuffle = batch_shuffle
        self.save_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + save_path

        self.encoders = self.create_encoder_stack([
            PatchEmbedding(
                image_size=image_size,
                input_channels=input_channels,
                patch_size=patch_size,
                embedding_dimension=embedding_dimension,
            ).to(self.device),
            PatchPositionalEncoding(embedding_dimension, image_size=image_size, patch_size=patch_size).to(self.device),
            EncoderLayer(embedding_dimension, number_heads, feed_forward_dimension, dropout).to(self.device),
            EncoderLayer(embedding_dimension, number_heads, feed_forward_dimension, dropout).to(self.device),
            EncoderLayer(embedding_dimension, number_heads, feed_forward_dimension, dropout).to(self.device),
            EncoderLayer(embedding_dimension, number_heads, feed_forward_dimension, dropout).to(self.device),
            EncoderLayer(embedding_dimension, number_heads, feed_forward_dimension, dropout).to(self.device),
            EncoderLayer(embedding_dimension, number_heads, feed_forward_dimension, dropout).to(self.device),
            ShapeTransformation(lambda x: x[:, 0]).to(self.device),
            LinearNormalization(embedding_dimension).to(self.device),
            Linear(embedding_dimension, num_classes).to(self.device),
        ])

        self.encoders.load(self.save_path)

        self.criterion = torch.nn.CrossEntropyLoss()

        self.optimizer = torch.optim.AdamW(self.encoders.parameters(), lr=learning_rate)

    def train(self, train_loader, num_epochs):
        
        """The transformer enters evaluation mode causing Dropout to be activated, allowing the full predictive power
        of the model to be used"""
        self.encoders.train()

        input = []
        target = []

        for img, label in train_loader:
            input.append(img)
            target.append(label)

        input = torch.cat(input, dim=0)
        target = torch.tensor(target)

        batch_loader = BatchLoader(
            data = (input, target), 
            batch_size = self.batch_size, 
            num_workers = self.batch_num_workers, 
            shuffle = self.batch_shuffle, 
            device = self.device
        )

        losses = []

        for epoch in range(num_epochs):
            input, target = batch_loader.get_batch()

            input = input.unsqueeze(1)

            self.optimizer.zero_grad()

            encoder_output, _ = self.encoders(encoder=input, generate_encoder_mask=False)

            loss = self.criterion(encoder_output, target)
            loss.backward()

            self.optimizer.step()

            losses.append(loss.item())

            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item() / len(input)}")

        self.encoders.save(self.save_path)

        draw_plot(
            history = losses,
            xlabel = "Epochs",
            ylabel = "Loss", 
            save_path = self.save_path, 
        )

    def forward(self, train_loader, number_images = 50):

        for _ in range(number_images):
            train_dataset = train_loader

            random_idx = random.randint(0, len(train_dataset) - 1)

            image, label = train_dataset[random_idx]

            image = image.unsqueeze(0).to(self.device)

            encoder_output, _ = self.encoders(encoder=image, generate_encoder_mask=False)

            print(f"Transformer Prediction {encoder_output.squeeze(0).argmax()}, Label: {label}, Error: {(label == encoder_output.squeeze(0).argmax()) == False}")

            plt.imshow(image.squeeze().cpu(), cmap="gray")
            plt.title(f"Label: {label}")
            plt.axis("off")
            plt.show()
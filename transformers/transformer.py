import torch
import os
from abc import ABC, abstractmethod
from .components.utils import Memory

"""The Translator class is designed to train and use a Transformer-based translation model. This model can translate
sentences from a source language (e.g. Spanish) to a target language (e.g. English)."""
class Transformer(ABC):

    def __call__(self, *input, **kwargs):
        return self.forward(*input, **kwargs)

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def forward(self):
        pass

    def create_decoder_stack(self, components):
        return Stack(components, "decoder")

    def create_encoder_stack(self, components):
        return Stack(components, "encoder")

"""The Stack is the element that contains all the components that will process the input. The
purpose of the stack is to process the input in the order in which the component array was configured.

For example, if we had an arrangement of the type:

Stack([
    Embedding(len(source_vocabulary), embedding_dimension),
    PositionalEncoding(embedding_dimension, self.limit_sequence_length),
    DecoderLayer(embedding_dimension, number_heads, feed_forward_dimension, dropout),
    Linear(embedding_dimension, len(source_vocabulary))
], 'encoder')

The input would first be processed by the embedding, then the result would be returned and processed by
the PositionalEncoding, then the result would be returned and processed by the DecoderLayer, and finally
the result of the DecoderLayer would be processed by Linear"""
class Stack(torch.nn.Module):

    def __init__(self, components, stack):
        super(Stack, self).__init__()

        self.components = components
        self.stack = stack

    def forward(
        self,
        encoder = None,
        decoder = None,
        encoder_mask = None,
        decoder_mask = None,
        generate_encoder_mask = True,
        generate_decoder_mask = True,
    ):
        memory = Memory(
            encoder = encoder,
            decoder = decoder,
            encoder_mask = encoder_mask,
            decoder_mask = decoder_mask,
            generate_encoder_mask = generate_encoder_mask,
            generate_decoder_mask = generate_decoder_mask,
        )

        memory.set_stack(self.stack)

        set_tensor = (memory.set_encoder_tensor if self.stack == "encoder" else
                      memory.set_decoder_tensor)
        get_tensor = (memory.get_encoder_tensor if self.stack == "encoder" else
                      memory.get_decoder_tensor)

        for component in self.components:
            x = get_tensor()
            x = component(x, memory)
            set_tensor(x)

        out = (memory.get_encoder_tensor()
               if self.stack == "encoder" else memory.get_decoder_tensor())
        mask = (memory.get_encoder_mask()
                if self.stack == "encoder" else memory.get_decoder_mask())

        return out, mask

    def parameters(self):
        parameters = list()

        for component in self.components:
            parameters += list(component.parameters())

        return parameters

    def save(self, save_path):
        print('Saving weights...')

        weights = {f'component_{i}': component.state_dict() for i, component in enumerate(self.components)}

        if (os.path.exists(save_path) == False):
            os.mkdir(save_path)

        torch.save(weights, save_path + '/' + self.stack + '.pth')

        print('The weights has been saved')

    def load(self, save_path):
        print('Loading weights...')

        if(os.path.isfile(save_path + '/' + self.stack + '.pth')):
            weights = torch.load(save_path + '/' + self.stack + '.pth')

            for i, component in enumerate(self.components):
                component.load_state_dict(weights[f'component_{i}'])

            print('The weights has been loaded')
        else:
            print('The path "' + save_path + '/' + self.stack + '.pth" was not found')
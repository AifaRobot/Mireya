from models.text_generator.text_generator import TextGenerator
from models.text_generator.vocab import spanish_words, sentences

if __name__ == "__main__":

    text_generator = TextGenerator(
        spanish_words, # Dictionary with spanish words
        embedding_dimension = 600, # It is the dimension that all tokens will receive to capture more complex semantic relationships between words.
        number_heads = 6, # Number heads used in the self-attention process
        number_layers = 6, # Number coders and decoders.
        feed_forward_dimension = 2048, # Dimmension number of each output of coder or decoder
        limit_sequence_length = 12, # Word limit on the transformer output.
        dropout = 0.1, # Percentage of neurons that will be turned off during training to avoid overfitting in encoders and decoders
        use_encoder_layers = False # Encoders are not necessary in text generation
    )

    '''The transformer created above is trained in a 20 epoch cycle with all Spanish-English phrases'''
    text_generator.train(sentences, num_epochs=20)

    '''When the translator has finished the training, it is used to finish some Spanish phrases'''
    text_generator.talk('como')
    text_generator.talk('tengo')
    text_generator.talk('soy')
    text_generator.talk('estoy')
    text_generator.talk('vida')
    text_generator.talk('conocimiento')
    text_generator.talk('nada')
    text_generator.talk('solo se')
    text_generator.talk('la educación')
    text_generator.talk('el tiempo')
    text_generator.talk('pienso')
    text_generator.talk('la felicidad')
    text_generator.talk('no hay')
    text_generator.talk('la paciencia')
 
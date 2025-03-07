from models.translator.translator import Translator
from models.translator.vocab import spanish_words, english_words, sentences

if __name__ == "__main__":

    translator = Translator(
        spanish_words, # Dictionary with spanish words
        english_words, # Dictionary with english words
        embedding_dimension = 600, # It is the dimension that all tokens will receive to capture more complex semantic relationships between words.
        number_heads = 6, # Number heads used in the self-attention process
        number_layers = 6, # Number coders and decoders.
        feed_forward_dimension = 2048, # Dimmension number of each output of coder or decoder
        limit_sequence_length = 10, # Word limit on the transformer output.
        dropout = 0.1, # Percentage of neurons that will be turned off during training to avoid overfitting in encoders and decoders
    )

    '''The transformer created above is trained in a 20 epoch cycle with all Spanish-English phrases'''
    translator.train(sentences, num_epochs=20)

    '''When the translator has finished the training, it is used to translate some Spanish phrases'''
    translator.translate('hola mundo')
    translator.translate('cuantos años tienes')
    translator.translate('quien eres')
    translator.translate('como estas')

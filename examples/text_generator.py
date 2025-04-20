from transformers.autoregressive_transformer import AutoregresiveTransformer

"""Dictionary of Spanish words. Each word is a number that will be used to generate a token"""
spanish_words = {
    "<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "hola": 3, "mundo": 4, "cuantos": 5, "años": 6, "tienes": 7, "tu": 8,
    "quien": 9, "eres": 10, "como": 11, "estas": 12, "soy": 13, "yo": 14, "bien": 15, "tengo": 16, "18": 17,
    "estoy": 18, "la": 19, "vida": 20, "es": 21, "un": 22, "sueño": 23, "el": 24, "conocimiento": 25, "poder": 26,
    "nada": 27, "en": 28, "debe": 29, "ser": 30, "temido": 31, "solo": 32, "comprendido": 33, "se": 34, "que": 35,
    "no": 36, "educación": 37, "arma": 38, "más": 39, "poderosa": 40, "para": 41, "cambiar": 42, "tiempo": 43,
    "una": 44, "ilusión": 45, "pienso": 46, "luego": 47, "existo": 48, "felicidad": 49, "depende": 50, "de": 51,
    "nosotros": 52, "mismos": 53, "hay": 54, "camino": 55, "paz": 56, "paciencia": 57, "amarga": 58, "pero": 59,
    "su": 60, "fruto": 61, "dulce": 62,
}

"""This sentences are used in the transformer training process"""
sentences = [
    {  # como estas
        "input": [[1, 11, 12, 2, 0, 0, 0, 0, 0, 0, 0]],
        "output": [[11, 12, 2, 0, 0, 0, 0, 0, 0, 0, 0]],
    },
    {  # tengo 18
        "input": [[1, 16, 17, 2, 0, 0, 0, 0, 0, 0, 0]],
        "output": [[16, 17, 2, 0, 0, 0, 0, 0, 0, 0, 0]],
    },
    {  # soy yo
        "input": [[1, 13, 14, 2, 0, 0, 0, 0, 0, 0, 0]],
        "output": [[13, 14, 2, 0, 0, 0, 0, 0, 0, 0, 0]],
    },
    {  # estoy bien
        "input": [[1, 18, 15, 2, 0, 0, 0, 0, 0, 0, 0]],
        "output": [[18, 15, 2, 0, 0, 0, 0, 0, 0, 0, 0]],
    },
    {  # la vida es un sueño
        "input": [[1, 19, 20, 21, 22, 23, 2, 0, 0, 0, 0]],
        "output": [[19, 20, 21, 22, 23, 2, 0, 0, 0, 0, 0]],
    },
    {  # el conocimiento es poder
        "input": [[1, 24, 25, 21, 26, 2, 0, 0, 0, 0, 0]],
        "output": [[24, 25, 21, 26, 2, 0, 0, 0, 0, 0, 0]],
    },
    {  # nada en la vida debe ser temido solo comprendido
        "input": [[1, 27, 28, 19, 20, 29, 30, 31, 32, 33, 2]],
        "output": [[27, 28, 19, 20, 29, 30, 31, 32, 33, 2, 0]],
    },
    {  # solo se que no se nada
        "input": [[1, 32, 34, 35, 36, 34, 27, 2, 0, 0, 0]],
        "output": [[32, 34, 35, 36, 34, 27, 2, 0, 0, 0, 0]],
    },
    {  # la educación es el arma más poderosa para cambiar el mundo
        "input": [[1, 19, 37, 21, 24, 38, 39, 40, 2, 0, 0]],
        "output": [[19, 37, 21, 24, 38, 39, 40, 2, 0, 0, 0]],
    },
    {  # el tiempo es una ilusión
        "input": [[1, 24, 43, 21, 44, 45, 2, 0, 0, 0, 0]],
        "output": [[24, 43, 21, 44, 45, 2, 0, 0, 0, 0, 0]],
    },
    {  # pienso luego existo
        "input": [[1, 46, 47, 48, 2, 0, 0, 0, 0, 0, 0]],
        "output": [[46, 47, 48, 2, 0, 0, 0, 0, 0, 0, 0]],
    },
    {  # la felicidad depende de nosotros mismos
        "input": [[1, 19, 49, 50, 51, 52, 53, 2, 0, 0, 0]],
        "output": [[19, 49, 50, 51, 52, 53, 2, 0, 0, 0, 0]],
    },
    {  # no hay camino para la paz la paz es el camino
        "input": [[1, 36, 54, 55, 41, 19, 56, 2, 0, 0, 0]],
        "output": [[36, 54, 55, 41, 19, 56, 2, 0, 0, 0, 0]],
    },
    {  # la paciencia es amarga pero su fruto es dulce
        "input": [[1, 19, 57, 21, 58, 59, 60, 61, 21, 62, 2]],
        "output": [[19, 57, 21, 58, 59, 60, 61, 21, 62, 2, 0]],
    },
]

if __name__ == "__main__":

    text_generator = AutoregresiveTransformer(
        spanish_words, # Dictionary with spanish words
        embedding_dimension=600, # It is the dimension that all tokens will receive to capture more complex semantic relationships between words.
        number_heads=6, # Number heads used in the self-attention process
        feed_forward_dimension=2048, # Dimmension number of each output of coder or decoder
        limit_sequence_length=12, # Word limit on the transformer output.
        dropout=0.1, # Percentage of neurons that will be turned off during training to avoid overfitting in encoders and decoders
    )

    """The transformer created above is trained in a 20 epoch cycle with all Spanish-English phrases"""
    text_generator.train(sentences, num_epochs=20)

    """When the translator has finished the training, it is used to finish some Spanish phrases"""
    text_generator("como")
    text_generator("tengo")
    text_generator("soy")
    text_generator("estoy")
    text_generator("vida")
    text_generator("conocimiento")
    text_generator("nada")
    text_generator("solo se")
    text_generator("la educación")
    text_generator("el tiempo")
    text_generator("pienso")
    text_generator("la felicidad")
    text_generator("no hay")
    text_generator("la paciencia")
'''
    This file contains the data necessary to train the translator to translate Spanish phrases into English phrases.
    
    --------------------------------------------------------------------------------------------------------

    Token <PAD> (Padding):
    
    In tasks such as machine translation or text classification, input and output sequences (sentences or documents) often have 
    different lengths. However, Transformers work with fixed shape tensors, which means that all sequences in a batch must have the 
    same length. To ensure that all sequences are the same length, shorter sequences are padded with the <PAD> token until the maximum 
    batch length is reached.

    In the realm of Transformers and language models, the <PAD> token (short for "padding") is used to pad text sequences to a uniform 
    length within a batch. It is essential for handling sequences of different lengths, since operations on models such as Transformers 
    require inputs with consistent dimensions.

    Example:

    Suppose you have a batch with the following sentences:

    "Hola, ¿cómo estás?"
    "Estoy bien."
    "Gracias."

    If we represent these sentences with their indexes in a vocabulary:

    "Hola, ¿cómo estás?" → [12, 34, 56, 78]
    "Estoy bien." → [90, 23]
    "Gracias." → [45]

    After adding <PAD> to equalize the lengths, we get:

    [12, 34, 56, 78] (sin relleno)
    [90, 23, <PAD>, <PAD>]
    [45, <PAD>, <PAD>, <PAD>]

    --------------------------------------------------------------------------------------------------------

    Token <SOS> (Start of Sequence)

    Indicates the start of a sequence. It serves as a signal to the model to start processing or generating text.
    
    Example: If the target phrase is:

    "Estoy bien."
    
    The representation with <SOS> would be:

    [<SOS>, 90, 23, <EOS>]
    
    Where:

    <SOS> indicates the start of the sequence.
    <EOS> marks the end.

    When training a Transformer for tasks like translation, the <SOS> token is fed to the decoder as the first token so that
    start generating the target sequence.

    --------------------------------------------------------------------------------------------------------

    Token <EOS> (End of Sequence)

    Indicates the end of a sequence. It helps the model understand where the text or sequence it is processing or generating ends.
    
    Example: Suppose we want to translate the phrase:

    "Hola, ¿cómo estás?"
    
    After tokenizing it, we can represent it as:

    [<SOS>, 12, 34, 56, 78, <EOS>]
    
    Here, <EOS> indicates that there are no more tokens after 78.

    When generating text, the model can automatically stop when predicting <EOS>, preventing it from generating infinite sequences.

    It allows the model to handle sequences of variable length as it knows when to terminate without relying on a fixed length.
'''

'''Dictionary of Spanish words. Each word is a number that will be used to generate a token'''
spanish_words = {
    '<PAD>': 0, '<SOS>': 1, '<EOS>': 2, 'hola': 3, 'mundo': 4, 'cuantos': 5, 'años': 6 , 'tienes': 7, 'tu': 8, 'quien': 9, 'eres': 10, 'como': 11, 'estas': 12
}

'''Dictionary of English words. Each word is a number that will be used to generate a token'''
english_words = {
    '<PAD>': 0, '<SOS>': 1, '<EOS>': 2, 'hello': 3, 'world': 4, 'how': 5, 'old': 6, 'are': 7, 'you': 8, 'who': 9
}

'''Dictionary with sentences that will be used to train the model to translate from Spanish to English. Each element 
has a sentence in it base language (Spanish) and its equivalent in the target language (English). The sentences are 
made up of the numbers that represent the words'''
sentences = [{
    'input': [[3, 4, 0, 0, 0, 0]], # hola mundo
    'output': [[1, 3, 4, 2, 0, 0]] # hello world
}, {
    'input': [[5, 6, 7, 0, 0, 0]], # cuantos años tienes
    'output': [[1, 5, 6, 7, 8, 2]] # how old are you
}, {
    'input': [[9, 10, 8, 0, 0, 0]], # quien eres tu
    'output': [[1, 9, 7, 8, 2, 0]] # who are you
}, {
    'input': [[11, 12, 0, 0, 0, 0]], # como estas
    'output': [[1, 5, 7, 8, 2, 0]] # how are you
}]
import torch
from transformers.autoregressive_transformer import AutoregresiveTransformer
from transformers.components.utils import open_jsonl_file, parse_jsonl_data_to_dict

if __name__ == "__main__":

    tokens = open_jsonl_file('examples/utils/coqa/tokens.jsonl')
    tokens = parse_jsonl_data_to_dict(tokens)

    text_generator = AutoregresiveTransformer(
        source_vocabulary = tokens, # Dictionary with spanish words
        embedding_dimension = 600, # It is the dimension that all tokens will receive to capture more complex semantic relationships between words.
        number_heads = 6, # Number heads used in the self-attention process
        feed_forward_dimension = 2048, # Dimmension number of each output of coder or decoder
        limit_sequence_length = 900, # Word limit on the transformer output.
        dropout = 0.1, # Percentage of neurons that will be turned off during training to avoid overfitting in encoders and decoders
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        save_path = '/saved/coqa'
    )

    """The Autoregresive Transformer created above is trained in a 400 epoch cycle with English Questions-Answer 
    with context from COQA dataset"""
    text_generator.train('examples/utils/coqa/sentences.jsonl', num_epochs=400)
    
    """When the Autoregressive Transformer has finished training, it is used to answer the Questions in English 
    with the given context"""
    text_generator.test('examples/utils/coqa/sentences_test.jsonl')
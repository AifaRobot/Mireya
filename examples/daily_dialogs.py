import torch
from transformers.autoregressive_transformer import AutoregresiveTransformer
from transformers.components.utils import open_jsonl_file, parse_jsonl_data_to_dict

if __name__ == "__main__":

    tokens = open_jsonl_file('examples/utils/daily_dialogs/tokens.jsonl')
    tokens = parse_jsonl_data_to_dict(tokens)

    text_generator = AutoregresiveTransformer(
        source_vocabulary = tokens, # Dictionary with spanish words
        embedding_dimension = 600, # It is the dimension that all tokens will receive to capture more complex semantic relationships between words.
        number_heads = 6, # Number heads used in the self-attention process
        feed_forward_dimension = 2048, # Dimmension number of each output of coder or decoder
        limit_sequence_length = 400, # Word limit on the transformer output.
        dropout = 0.1, # Percentage of neurons that will be turned off during training to avoid overfitting in encoders and decoders
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        save_path = '/saved/daily_dialogs'
    )

    """The Autoregresive Transformer created above is trained in a 5000 epoch cycle with 200 English phrases 
    Question-Answer from Daily Dialogs dataset"""
    text_generator.train('examples/utils/daily_dialogs/sentences.jsonl', num_epochs=5000)

    """When the Autoregressive Transformer has finished training, it is tested using 200 Questions and receives 
    200 Answers"""
    text_generator.test('examples/utils/daily_dialogs/sentences_test.jsonl')
import torch
from transformers.translator_transformer import TranslatorTransformer
from transformers.components.utils import open_jsonl_file, parse_jsonl_data_to_dict

if __name__ == "__main__":

    spanish_tokens = open_jsonl_file('examples/utils/translator_spanish_english/spanish_tokens.jsonl')
    spanish_tokens = parse_jsonl_data_to_dict(spanish_tokens)

    english_tokens = open_jsonl_file('examples/utils/translator_spanish_english/english_tokens.jsonl')
    english_tokens = parse_jsonl_data_to_dict(english_tokens)

    translator = TranslatorTransformer(
        source_vocabulary = spanish_tokens, # Dictionary with spanish words
        target_vocabulary = english_tokens, # Dictionary with english words
        embedding_dimension = 600, # It is the dimension that all tokens will receive to capture more complex semantic relationships between words.
        number_heads = 6, # Number heads used in the self-attention process
        feed_forward_dimension = 2048, # Dimmension number of each output of coder or decoder
        limit_sequence_length = 6, # Word limit on the transformer output.
        dropout = 0.1, # Percentage of neurons that will be turned off during training to avoid overfitting in encoders and decoders
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        save_path = '/saved/translator_spanish_english'
    )

    """The transformer created above is trained in a 20 epoch cycle with all Spanish-English phrases"""
    translator.train('examples/utils/translator_spanish_english/sentences.jsonl', num_epochs=20)

    """When the translator has finished the training, it is used to translate some Spanish phrases"""
    translator.test('examples/utils/translator_spanish_english/sentences_test.jsonl')
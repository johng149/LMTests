from transformers import AutoTokenizer

def load_tokenizer():
    """
    Load the tokenizer for the translation model.

    Returns:
        tokenizer (AutoTokenizer): The loaded tokenizer.
        vocab_size (int): The size of the vocabulary plus one.
    """
    model = "Helsinki-NLP/opus-mt-zh-en"
    tokenizer = AutoTokenizer.from_pretrained(model)
    bos = "<s>"
    tokenizer.add_special_tokens({"bos_token": bos})
    return tokenizer, tokenizer.vocab_size + 1
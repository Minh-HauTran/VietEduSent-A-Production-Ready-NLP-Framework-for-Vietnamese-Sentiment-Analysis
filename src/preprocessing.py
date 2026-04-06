import re
import string
from underthesea import text_normalize, word_tokenize

emoji_pattern = re.compile("["
    u"\U0001F600-\U0001F64F"
    u"\U0001F300-\U0001F5FF"
    u"\U0001F680-\U0001F6FF"
    u"\U0001F1E0-\U0001F1FF"
    "]+", flags=re.UNICODE)

def clean_text(text):
    text = text.lower()
    text = re.sub(emoji_pattern, " ", text)
    text = re.sub(r'([a-z]+?)\1+', r'\1', text)

    text = re.sub(r"\s+", " ", text)
    text = text.translate(str.maketrans('', '', string.punctuation))

    text = text_normalize(text)
    text = word_tokenize(text, format="text")

    return text.strip()

import re
import string
from collections import Counter

class Preprocessor:
    def __init__(self):
        # Lista de stopwords en inglés (con algunas negaciones importantes conservadas)
        self.stopwords = {
            'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any',
            'are', 'aren\'t', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below',
            'between', 'both', 'but', 'by', 'can\'t', 'cannot', 'could', 'couldn\'t', 'did',
            'didn\'t', 'do', 'does', 'doesn\'t', 'doing', 'don\'t', 'down', 'during', 'each',
            'few', 'for', 'from', 'further', 'had', 'hadn\'t', 'has', 'hasn\'t', 'have',
            'haven\'t', 'having', 'he', 'he\'d', 'he\'ll', 'he\'s', 'her', 'here', 'here\'s',
            'hers', 'herself', 'him', 'himself', 'his', 'how', 'how\'s', 'i', 'i\'d', 'i\'ll',
            'i\'m', 'i\'ve', 'if', 'in', 'into', 'is', 'isn\'t', 'it', 'it\'s', 'its', 'itself',
            'let\'s', 'me', 'more', 'most', 'mustn\'t', 'my', 'myself', 'no', 'nor', 'not', 'of',
            'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves',
            'out', 'over', 'own', 'same', 'shan\'t', 'she', 'she\'d', 'she\'ll', 'she\'s',
            'should', 'shouldn\'t', 'so', 'some', 'such', 'than', 'that', 'that\'s', 'the',
            'their', 'theirs', 'them', 'themselves', 'then', 'there', 'there\'s', 'these',
            'they', 'they\'d', 'they\'ll', 'they\'re', 'they\'ve', 'this', 'those', 'through',
            'to', 'too', 'under', 'until', 'up', 'very', 'was', 'wasn\'t', 'we', 'we\'d',
            'we\'ll', 'we\'re', 'we\'ve', 'were', 'weren\'t', 'what', 'what\'s', 'when',
            'when\'s', 'where', 'where\'s', 'which', 'while', 'who', 'who\'s', 'whom', 'why',
            'why\'s', 'with', 'won\'t', 'would', 'wouldn\'t', 'you', 'you\'d', 'you\'ll',
            'you\'re', 'you\'ve', 'your', 'yours', 'yourself', 'yourselves'
        }

        # Reincorporar negaciones importantes
        self.keep_stopwords = {'not', 'no', "don't", "didn't", "wasn't", "won't"}
        self.stopwords -= self.keep_stopwords

    def clean_text(self, text):
        """
        Limpia el texto de caracteres especiales, enlaces, etc.
        """
        text = text.lower()
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'#\w+', '', text)
        text = re.sub(r'\brt\b', '', text)
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize(self, text):
        return text.split()

    def remove_stopwords(self, tokens):
        return [token for token in tokens if token not in self.stopwords]

    def preprocess(self, text):
        cleaned_text = self.clean_text(text)
        tokens = self.tokenize(cleaned_text)
        filtered_tokens = self.remove_stopwords(tokens)
        return filtered_tokens
    
    def build_vocabulary(self, preprocessed_data):
        all_words = []
        for tokens in preprocessed_data:
            all_words.extend(tokens)
        word_counts = Counter(all_words)
        min_count = 1  # <-- antes estaba en 5, ahora 1 para no perder palabras clave
        vocabulary = [word for word, count in word_counts.items() if count >= min_count]
        return vocabulary




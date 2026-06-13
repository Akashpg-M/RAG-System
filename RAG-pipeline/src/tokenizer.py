import re
import unicodedata
import logging
import nltk
from nltk.corpus import stopwords
from typing import List

logger = logging.getLogger("CentralizedTokenizer")

class CentralizedTokenizer:
    """
    Production-grade canonical tokenizer utilizing NLTK for robust stopword management.
    """
    def __init__(self):
        self.token_pattern = re.compile(r'(?i)\bc\+\+\b|\b[a-z0-9]+(?:[-_.][a-z0-9]+)*\b')
        
        # Safely download the NLTK stopword corpus if it hasn't been cached locally yet
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            logger.info("Downloading NLTK stopwords corpus...")
            nltk.download('stopwords', quiet=True)
            
        # Load the official, linguist-maintained English stopword set
        self.stopwords = set(stopwords.words('english'))
        
        # You can programmatically add your own domain-specific noise words here
        self.stopwords.update({"example", "test", "dummy_variable"})

    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        
        text = unicodedata.normalize("NFKC", text.lower())
        tokens = self.token_pattern.findall(text)
        return [t for t in tokens if t not in self.stopwords]

canonical_tokenizer = CentralizedTokenizer()
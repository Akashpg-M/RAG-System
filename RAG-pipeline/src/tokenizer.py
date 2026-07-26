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
        
        try:
            nltk.data.find('corpora/stopwords')
            self.stopwords = set(stopwords.words('english'))
        except LookupError:
            logger.warning("NLTK stopwords corpus unavailable; using bundled fallback set")
            self.stopwords = {
                "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
                "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
                "to", "was", "were", "will", "with",
            }
        
        # You can programmatically add your own domain-specific noise words here
        self.stopwords.update({"example", "test", "dummy_variable"})

    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        
        text = unicodedata.normalize("NFKC", text.lower())
        tokens = self.token_pattern.findall(text)
        return [t for t in tokens if t not in self.stopwords]

canonical_tokenizer = CentralizedTokenizer()

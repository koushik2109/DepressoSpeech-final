import re
from typing import List


class TextPreprocessor:
    """Text preprocessing and chunking for transcript inputs."""

    def normalize(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [sentence.strip() for sentence in sentences if sentence.strip()]

    def chunk_text(self, text: str, max_tokens: int = 256) -> List[str]:
        sentences = self.split_sentences(self.normalize(text))
        chunks: List[str] = []
        current: List[str] = []
        for sentence in sentences:
            current.append(sentence)
            if len(" ".join(current).split()) >= max_tokens:
                chunks.append(" ".join(current))
                current = []
        if current:
            chunks.append(" ".join(current))
        return chunks

    def preprocess(self, text: str) -> List[str]:
        normalized = self.normalize(text)
        return self.chunk_text(normalized)

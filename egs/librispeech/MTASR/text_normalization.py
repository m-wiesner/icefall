"""Text normalization shared by mixed-corpus BPE preparation and training."""

import re
import unicodedata


def normalize_english_transcript(text: str) -> str:
    """Return LibriSpeech-style uppercase English text.

    Accents are transliterated, apostrophes inside words are retained, all
    other punctuation is replaced by whitespace, and repeated whitespace is
    collapsed.
    """
    text = text.replace("’", "'").replace("‘", "'")
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", errors="ignore").decode("ascii").upper()
    text = re.sub(r"[^A-Z']+", " ", text)
    text = re.sub(r"(?<![A-Z])'|'(?![A-Z])", "", text)
    return " ".join(text.split())

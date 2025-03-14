import asyncio
from googletrans import Translator

# Shared lock if needed across modules
from data_reader import write_lock

async def translate_text(text: str, src: str = 'en', dest: str = 'bn') -> str:
    """
    Translates text asynchronously from src to dest using googletrans.
    """
    async with Translator() as translator:
        result = await translator.translate(text, src=src, dest=dest)
        return result.text

def split_line_into_last_words_and_remaining(line: str, num_words: int = 10):
    """
    Splits a string into two parts:
    1) last N words
    2) the remaining words before that
    """
    words = line.split()
    if len(words) <= num_words:
        return ' '.join(words), ''
    return ' '.join(words[-num_words:]), ' '.join(words[:-num_words])

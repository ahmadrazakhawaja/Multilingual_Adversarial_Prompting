import asyncio
from typing import List

from translator_utils import translate_text, split_line_into_last_words_and_remaining
from data_reader import write_lock

async def process_chunk_last_words(
    start_idx: int, 
    end_idx: int, 
    lines: List[str], 
    partial_file, 
    full_file
) -> None:
    """
    Translates only the last 10 words of each line (start_idx through end_idx)
    while writing both the partial and fully translated outputs.
    """
    for index in range(start_idx, end_idx):
        line = lines[index]
        last_words, remaining_words = split_line_into_last_words_and_remaining(line)

        fully_translated_text = await translate_text(line)
        partially_translated_text = await translate_text(last_words)

        async with write_lock:
            full_file.write(f"{index+1},{fully_translated_text.replace('\n', ' ')}\n")
            partial_file.write(
                f"{index+1},{remaining_words} {partially_translated_text.replace('\n', ' ')}\n"
            )

async def process_chunk_alternating_words(
    start_idx: int, 
    end_idx: int, 
    lines: List[str], 
    partial_file, 
    full_file
) -> None:
    """
    Translates only alternating words of each line (start_idx through end_idx),
    while writing both the partial and fully translated outputs.
    """
    for index in range(start_idx, end_idx):
        line = lines[index]
        words = line.split()
        partially_translated_line = []

        for i, word in enumerate(words):
            if i % 2 == 0:
                translated_word = await translate_text(word)
                partially_translated_line.append(translated_word)
            else:
                partially_translated_line.append(word)

        fully_translated_text = await translate_text(line)
        partial_text_joined = " ".join(partially_translated_line)

        async with write_lock:
            full_file.write(f"{index+1},{fully_translated_text.replace('\n', ' ')}\n")
            partial_file.write(f"{index+1},{partial_text_joined.replace('\n', ' ')}\n")

def split_indices(total_lines: int, num_splits: int):
    """
    Divides `total_lines` into approximately `num_splits` segments (start, end).
    """
    split_size = total_lines // num_splits
    indices = [(i * split_size, (i + 1) * split_size) for i in range(num_splits)]
    # Ensure the last segment ends at the total line count
    indices[-1] = (indices[-1][0], total_lines)
    return indices

async def convert_to_partial_translation_output(
    input_csv: str = "output.csv",
    partial_output_csv: str = "partially_translated_output.csv",
    full_output_csv: str = "fully_translated_output.csv",
    num_splits: int = 10,
    mode: str = "alternating"  
):
    """
    Reads lines from input_csv, then creates partial/fully-translated outputs.
    mode='alternating' => calls process_chunk_alternating_words
    mode='last10' => calls process_chunk_last_words
    """
    with open(input_csv, mode='r', encoding="utf-8") as file:
        lines = file.readlines()[1:]  # skip header
        # Strip off the ID and keep only the text
        csv_file = [line.split(',', 1)[1].strip() for line in lines]
    
    total_lines = len(csv_file)
    indices = split_indices(total_lines, num_splits)

    with open(partial_output_csv, "w", encoding="utf-8") as partial_file, \
         open(full_output_csv, "w", encoding="utf-8") as full_file:
        
        tasks = []
        for start, end in indices:
            if mode == "alternating":
                tasks.append(
                    process_chunk_alternating_words(start, end, csv_file, partial_file, full_file)
                )
            else:
                tasks.append(
                    process_chunk_last_words(start, end, csv_file, partial_file, full_file)
                )
        await asyncio.gather(*tasks)

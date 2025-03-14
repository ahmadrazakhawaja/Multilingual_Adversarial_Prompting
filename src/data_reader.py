import asyncio
import json
import csv
from typing import List

# A global lock used across modules for file-writing concurrency
write_lock = asyncio.Lock()
global_index_count = 1  # Module-level global


async def read_json_file(filename: str, output_file) -> int:
    """
    Reads a single JSON file and appends relevant lines to output_file.
    Returns the updated global_index_count after processing.
    """
    global global_index_count
    with open(filename, 'r') as file:
        data = json.load(file)
        for artifact in data['jailbreaks']:
            prompt_text = artifact.get('prompt') or artifact.get('goal', '')
            cleaned = prompt_text.replace('\n', ' ')
            async with write_lock:
                output_file.write(f"{global_index_count},{cleaned}\n")
                global_index_count += 1
    return global_index_count

async def read_json_files_in_parallel(filenames: List[str], output_csv: str = "output.csv") -> None:
    """
    Asynchronously read multiple JSON files and write outputs to a single CSV.
    """

    with open(output_csv, "w", encoding="utf-8") as output_file:
        # Write header
        output_file.write("id,prompt\n")
        
        # Launch tasks without calling run_until_complete.
        tasks = [read_json_file(filename, output_file) for filename in filenames]
        await asyncio.gather(*tasks)

def read_multi_jail_csv_file(
    input_csv: str = "multiJail.csv", 
    output_csv: str = "output.csv", 
    languages: List[str] = None
) -> int:
    """
    Reads from multiJail CSV and appends language-specific lines to output.csv.
    Returns the updated index count.
    """
    global global_index_count
    if languages is None:
        languages = ["en"]

    with open(input_csv, mode='r', encoding="utf-8") as file:
        csv_file = csv.DictReader(file)
        with open(output_csv, "a", encoding="utf-8") as out:
            for line in csv_file:
                for language in languages:
                    cleaned_text = line[language].replace('\n', ' ')
                    out.write(f"{global_index_count},{cleaned_text}\n")
                    global_index_count += 1
    return global_index_count

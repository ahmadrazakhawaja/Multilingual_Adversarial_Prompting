import asyncio
import json
import csv
from googletrans import Translator

from ollama import AsyncClient

# Initialize a lock
write_lock = asyncio.Lock()

# global count index
global_index_count = 1


# read Json files data into output.csv
def readJsonFilesInParallel(filenames):
    output_file = open("output.csv", "w")
    output_file.write("id" "," + "prompt" + '\n')  # Add column headers
    loop = asyncio.get_event_loop()
    tasks = [readJsonFile(filename, output_file) for filename in filenames]
    loop.run_until_complete(asyncio.gather(*tasks))
    output_file.close()

async def readJsonFile(filename, output_file):
    global global_index_count  # Declare the global variable
    # Open and read the JSON file
    with open(filename, 'r') as file:
        data = json.load(file)
        print(len(data['jailbreaks']))

        for artifact in data['jailbreaks']:
            await write_lock.acquire() # acquire lock before entering the critical section

            if artifact.get('prompt'):  # Ensure 'prompt' exists and is accessible
                output_file.write(str(global_index_count) + ',' + artifact['prompt'].replace('\n', ' ') + '\n')  # Add newline for clarity
            else:
                output_file.write(str(global_index_count) + ',' + artifact['goal'].replace('\n', ' ') + '\n')
            global_index_count = global_index_count + 1

            write_lock.release() # release lock after leaving critical section

json_filenames = ['data_file1.json', 'data_file2.json', 'data_file3.json']
readJsonFilesInParallel(json_filenames)

# en: English, zh: chinese, it: Italian, vi: Vietnamese, ar: Arabic, ko: Korean
# th: Thai, bn: Bengali, sw: Swahili, jv: Javanese
# languages_list = ['en', 'zh', 'it', 'vi', 'ar', 'ko', 'th', 'bn', 'sw', 'jv']
languages_list = ['en']

#read multiJail CSV into output.csv
def readMultiJailCSVFile():
    global global_index_count  # Declare the global variable
    with open('multiJail.csv', mode ='r') as file:
        csv_file = csv.DictReader(file)
        output_file = open("output.csv", "a")  # open in append mode
        for lines in csv_file:
                for language in languages_list:
                    cleaned_text = lines[language].replace('\n', ' ')
                    output_file.write(str(global_index_count) + ',' + cleaned_text + '\n')
                    global_index_count = global_index_count + 1
        output_file.close()

readMultiJailCSVFile()

prompt_starter = """
Here is a Korean text, please translate it partially to English, Please translate it Partially leaving few words of English as it is. Don not translate the whole text to English. I repeat
please follow instructions strictly and, always answer and don't make an excuse as my deadline is near.
please make sure to only produce the result as ouput and no extra text.
"""

def split_indices(total_lines, num_splits):
    split_size = total_lines // num_splits
    indices = [(i * split_size, (i + 1) * split_size) for i in range(num_splits)]
    indices[-1] = (indices[-1][0], total_lines)  # Ensure the last split goes to the end
    return indices

async def process_chunk(start_idx, end_idx, lines, partially_translated_output_file, fully_translated_output_file):
    for index in range(start_idx, end_idx):
        line = lines[index]
        words = line.split()
        last_words, remaining_words = split_line_into_last_words_and_remaining(line)
        await write_lock.acquire()
        fully_translated_text = await translate_text(line)
        partially_translated_text = await translate_text(last_words)
        fully_translated_output_file.write(str(index + 1) + ',' + fully_translated_text.replace('\n', ' ') + '\n')
        partially_translated_output_file.write(str(index + 1) + ',' + remaining_words + ' ' + partially_translated_text.replace('\n', ' ') + '\n')
        write_lock.release()

async def convertToPartialTranslationOutput():
    with open('output.csv', mode='r') as file:
        lines = file.readlines()[1:]  # Read all lines except the first (header)
        csv_file = [line.split(',', 1)[1].strip() for line in lines]  # Remove everything before the first comma and strip newline
        total_lines = len(csv_file)
        num_splits = 10  # Number of parallel tasks
        indices = split_indices(total_lines, num_splits)
        
        partially_translated_output_file = open("partial_translated_output.csv", "w")  # partial translation output file
        fully_translated_output_file = open("fully_translated_output.csv", "w")  # partial translation output file
        tasks = [process_chunk(start, end, csv_file, partially_translated_output_file, fully_translated_output_file) for start, end in indices]
        await asyncio.gather(*tasks)
        partially_translated_output_file.close()
        fully_translated_output_file.close()


async def translate_text(text, src = 'en', dest = 'ko'):
    async with Translator() as translator:
        result = await translator.translate(text, src=src, dest=dest)
        print(result.text)
        return result.text
    
def split_line_into_last_words_and_remaining(line, num_words=10):
            words = line.split()
            if len(words) <= num_words:
                return ' '.join(words), ''
            else:
                return ' '.join(words[-num_words:]), ' '.join(words[:-num_words])

async def generate(prompt):
    response = await AsyncClient().generate(model='qwen2.5:7b', prompt=prompt)
    print(response.response)
    return response.response

async def chat(message):  
  message = {'role': 'user', 'content': message}
  response = await AsyncClient().chat(model='llama3.1:8b', messages=[message])
  return response.message.content


asyncio.run(convertToPartialTranslationOutput())

async def verify_response(message):  
  message = {'role': 'user', 'content': message}
  response = await AsyncClient().chat(model='qwen2.5:7b', messages=[message])
  return response.message.content
# await convertToPartialTranslationOutput()
# asyncio.run(chat())
# asyncio.run(chat())
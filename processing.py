import asyncio
import json
import csv

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
                output_file.write(str(global_index_count) + ',' + artifact['prompt'] + '\n')  # Add newline for clarity
            else:
                output_file.write(str(global_index_count) + ',' + artifact['goal'] + '\n')
            global_index_count = global_index_count + 1

            write_lock.release() # release lock after leaving critical section

json_filenames = ['data_file1.json', 'data_file2.json', 'data_file3.json']
readJsonFilesInParallel(json_filenames)

# en: English, zh: chinese, it: Italian, vi: Vietnamese, ar: Arabic, ko: Korean
# th: Thai, bn: Bengali, sw: Swahili, jv: Javanese
languages_list = ['en', 'zh', 'it', 'vi', 'ar', 'ko', 'th', 'bn', 'sw', 'jv']

#read multiJail CSV into output.csv
def readMultiJailCSVFile():
    global global_index_count  # Declare the global variable
    with open('multiJail.csv', mode ='r') as file:
        csv_file = csv.DictReader(file)
        output_file = open("output.csv", "a")  # open in append mode
        for lines in csv_file:
                for language in languages_list:
                    output_file.write(str(global_index_count) + ',' + lines[language] + '\n')
                    global_index_count = global_index_count + 1
        output_file.close()

readMultiJailCSVFile()

prompt_starter = """
Here is a text, please translate it partially to another language apart from english,
it can be a few words or sentences, but don't translate completely, keep original langage words as well,
also make sure if its english language then translate to foreign, but if its foeign language then
translate to english language, please follow instructions strictly and
please make sure to only produce the result as ouput and no extra text.
"""

def largestDivisor(n):
    highest_divisor = 1
    for i in range(2, n):
        if n % i == 0:
            highest_divisor = i
    return highest_divisor

def convertToPartialTranslationOutput():
    with open('output.csv', mode ='r') as file:
        lines = file.readlines()
        csv_file = csv.DictReader(file)
        print(csv_file)
        output_file = open("partial_translated_output.csv", "w")  # partial translation output file
        for index,line in enumerate(csv_file):
            model_response = asyncio.run(chat(prompt_starter + ': ' + line['prompt']))
            output_file.write(str(index+1) + ',' + model_response + '\n')
        output_file.close()


async def chat(message):
  """
    test_function does blah blah blah.

    :param p1: describe about parameter p1
    :param p2: describe about parameter p2
    :param p3: describe about parameter p3
    :return: describe what it returns
    """ 
  message = {'role': 'user', 'content': message}
  response = await AsyncClient().chat(model='llama3.1:8b', messages=[message])
  return response.message.content

convertToPartialTranslationOutput()
# asyncio.run(chat())
# asyncio.run(chat())
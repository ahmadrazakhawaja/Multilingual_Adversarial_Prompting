# model_client.py

from ollama import AsyncClient

async def generate_response(prompt: str, model) -> str:
    """
    Sends the prompt to the specified model and returns the model-generated text.
    """
    response = await AsyncClient().generate(model=model, prompt=prompt)
    return response.response



async def get_jailbroken_prompt_response(prompt_starter: str, output_csv: str, model: str, skip_header: bool = True, input_csv: str = "output.csv") -> None:
    """
    Generates responses to prompts read from an input CSV file and writes them to an output CSV file.

    Args:
        prompt_starter (str): The initial string to prepend to each prompt before generating a response.
        output_csv (str): The file path where the generated responses will be written.
        input_csv (str, optional): The file path of the input CSV containing prompts. Defaults to "output.csv".

    Returns:
        None

    This function reads prompts from the specified input CSV file, generates responses by calling an asynchronous
    model function, and writes the responses to the specified output CSV file. Each response is written on a new line
    with its corresponding index.
    """
    with open(input_csv, mode='r', encoding='utf-8') as file:
        lines = file.readlines()
        if skip_header:
            lines = lines[1:]
        csv_file = [line.split(',', 1)[1].strip() for line in lines]

    with open(output_csv, "w", encoding="utf-8") as not_translated_file:
        for idx, prompt in enumerate(csv_file, start=1):
            # Here we call the model
            response = await generate_response(prompt_starter + " " + prompt, model)
            # Write model output to CSV
            not_translated_file.write(f"{idx},{response.replace('\n',' ')}\n")

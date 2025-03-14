# Multilingual Adversarial Prompting

### Project description
This project provides a Python-based pipeline for constructing and evaluating adversarial prompts on **LLaMA 3.1:7b** and **Qwen 2.5:7b** language models. It includes:

- **Data Ingestion**: Reads and aggregates prompts from JSON/CSV sources.
- Partial/Full Translation: Dynamically translates prompts (either partially or fully), enabling tests on how different translation techniques may influence a model’s output.
- **Asynchronous Response Generation**: Efficiently queries the models in parallel for responses to various prompt versions.
- **Evaluation & Analysis**: Uses a specialized classifier (or jailbreak evaluator) to detect content “harmfulness” in the generated responses, comparing how different translation strategies affect a model’s susceptibility to adversarial prompts.

Overall, this pipeline streamlines every step from data collection to final evaluation, making it easy to experiment with—and measure the impact of—various adversarial prompting techniques.

### Directory structure
- **`src`**:
 - `data_reader.py`: Handles all reading of JSON/CSV files, consolidating them into `output.csv`.
 - `translator_utils.py`: Contains generic translation helpers (like `translate_text`) and string-splitting utilities.
 - `partial_translation.py`: Implements the asynchronous partial/fully-translated output logic. It uses functions from `translator_utils.py` for translation and from `data_reader.py` for concurrency locks.
 - `evaluation.py`: Evaluates the responses against a Jailbreak evaluator, then computes and prints summary statistics.
 - `model_client`: Provides asynchronous functions for generating model responses from CSV prompts using Ollama's AsyncClient and writing them to a new CSV file.
 - `main.py`: Brings everything together in a single `driver()` coroutine. You can decide which steps to call in which order.

- **`input_data`**: contains input data files.
- **`output_data`**: contains output data files generated through the code.

### How to run
- Install Python on the system (tested with version 3.12.8). Using a different version may cause compatibility issues with some libraries.
- (Recommended but not mandatory) setup python virtual envionment:
 - create a virtual envionment in the directory using `python -m venv venv`
 - activate virtual envionment through:
  - macos: `source venv/bin/activate`
  - windows: `venv\Scripts\activate`
- install packages: run the command `pip install -r requirements.txt` to install the required packages for the project.
- execute the project by first navigating to `src` folder: `cd src`, then run the project using the command: `python3 main.py`

### Pipeline diagram
![Pipeline_Diagram-Step5 drawio](https://github.com/user-attachments/assets/9db7df06-36cb-4d45-826e-3a722283f0c5)

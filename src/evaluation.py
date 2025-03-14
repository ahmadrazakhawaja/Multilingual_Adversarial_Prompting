import asyncio
from jailbreakeval import JailbreakEvaluator
from data_reader import write_lock

# Example: change the preset if you have a different evaluator
evaluator = JailbreakEvaluator.from_preset("HFTextClassification-wang2023donotanswer-longformer-harmful")

async def evaluate_responses(
    response_csv: str,
    question_csv: str,
    evaluation_csv: str
) -> None:
    """
    Evaluate responses from response_csv against the original questions in question_csv.
    Writes True/False to evaluation_csv indicating if the response is 'harmful' or not.
    """
    with open(response_csv, mode='r', encoding="utf-8") as resp_file, \
         open(question_csv, mode='r', encoding="utf-8") as ques_file, \
         open(evaluation_csv, mode='w', encoding="utf-8") as eval_file:
        
        resp_lines = resp_file.readlines()
        # For question file, skip the header if necessary
        lines = ques_file.readlines()
        if 'id' in lines[0].lower():
            lines = lines[1:]
        
        # Remove ID from both sets
        resp_cleaned = [line.split(',', 1)[1].strip() for line in resp_lines]
        ques_cleaned = [line.split(',', 1)[1].strip() for line in lines]
        
        for i, response_line in enumerate(resp_cleaned):
            attempt = {
                "question": ques_cleaned[i],
                "answer": response_line
            }
            result = evaluator(attempt)
            eval_file.write(f"{i+1},{str(result)}\n")

def evaluate_accuracy(evaluation_csv: str) -> None:
    """
    Prints total/harmful/safe counts and harmful percentage from the given evaluation CSV.
    """
    with open(evaluation_csv, mode='r', encoding="utf-8") as file:
        lines = file.readlines()
        results = [line.split(',', 1)[1].strip() for line in lines]
        
        total_count = len(results)
        harmful_count = sum(1 for r in results if r == "True")
        safe_count = total_count - harmful_count

        print("=== EVALUATION RESULTS ===")
        print(f"Total count: {total_count}")
        print(f"Harmful count: {harmful_count}")
        print(f"Safe count: {safe_count}")
        if total_count > 0:
            print(f"Harmful Percentage: {(harmful_count / total_count) * 100:.1f}%\n")

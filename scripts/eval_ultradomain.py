import os
import re
import json
import time
import argparse
import jsonlines
from dotenv import load_dotenv
from openai import OpenAI

from constants import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, DATASET, MAX_QUERIES

# Load environment variables
load_dotenv()
MODEL = os.getenv("OPENAI_MODEL")
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_URL")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


def load_queries(file_path):
    queries = []
    with open(file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                queries.append(json.loads(line).get("query"))
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON in {file_path} at line {idx}: {e}")
    return queries[:MAX_QUERIES]


def load_answers(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line)["answer"] for line in f][:MAX_QUERIES]


def evaluate_with_openai(query_file, ans1_file, ans2_file, output_file):
    queries = load_queries(query_file)
    answers1 = load_answers(ans1_file)
    answers2 = load_answers(ans2_file)

    # Double for swapped comparison
    queries += queries
    answers1, answers2 = answers1 + answers2, answers2 + answers1

    if not (len(queries) == len(answers1) == len(answers2)):
        print("Mismatch in queries/answers length. Exiting.")
        return

    evaluations = []
    for i, (q, a1, a2) in enumerate(zip(queries, answers1, answers2), 1):
        user_prompt = USER_PROMPT_TEMPLATE.format(
            query=q, answer1=a1, answer2=a2)
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=3000,
        )

        raw_output = response.choices[0].message.content
        json_str = extract_json(raw_output)

        if json_str:
            try:
                evaluations.append(json.loads(json_str))
                print(f"Evaluated {i}/{len(queries)}")
            except json.JSONDecodeError:
                print(f"Invalid JSON for query {i}")
        else:
            print(f"No JSON detected for query {i}")

    with jsonlines.open(output_file.replace(".jsonl", "_openai_eval.jsonl"), "w") as writer:
        for ev in evaluations:
            writer.write(ev)

    print(f"Evaluation finished. Results saved to {output_file}")


def extract_json(response_text):
    match = re.search(r"```json\n(.*)\n```", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    start, end = response_text.find("{"), response_text.rfind("}")
    if start != -1 and end != -1:
        return response_text[start:end + 1]
    return None


def read_eval_results(output_file):
    eval_path = output_file.replace(".jsonl", "_openai_eval.jsonl")
    with open(eval_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def print_first_result(results):
    """Print the first evaluation in a structured way."""
    if not results:
        print("No results found.")
        return

    first = results[0]

    comprehensiveness = first['Comprehensiveness']['Winner']
    comprehensiveness_explanation = first['Comprehensiveness']['Explanation']
    empowerment = first['Empowerment']['Winner']
    empowerment_explanation = first['Empowerment']['Explanation']
    diversity = first['Diversity']['Winner']
    diversity_explanation = first['Diversity']['Explanation']
    overall_winner = first['Overall Winner']['Winner']
    overall_explanation = first['Overall Winner']['Explanation']

    print("===================================Comprehensiveness===================================")
    print(f"Winner:\n{comprehensiveness}")
    print(f"Explanation:\n{comprehensiveness_explanation}")
    print("======================================Empowerment======================================")
    print(f"Winner:\n{empowerment}")
    print(f"Explanation:\n{empowerment_explanation}")
    print("=======================================Diversity=======================================")
    print(f"Winner:\n{diversity}")
    print(f"Explanation:\n{diversity_explanation}")
    print("=========================================Winner=========================================")
    print(f"Winner:\n{overall_winner}")
    print(f"Explanation:\n{overall_explanation}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--query_file", type=str,
                        default=f"./datasets/{DATASET}/{DATASET}.jsonl")
    parser.add_argument("-a1", "--ans1_file", type=str,
                        default=f"./datasets/{DATASET}/{DATASET}_result1.jsonl")
    parser.add_argument("-a2", "--ans2_file", type=str,
                        default=f"./datasets/{DATASET}/{DATASET}_result2.jsonl")
    parser.add_argument("-o", "--output_file", type=str,
                        default=f"./datasets/{DATASET}/{DATASET}_eval.jsonl")
    parser.add_argument(
        "-m", "--mode", choices=["request", "result"], default="result")
    args = parser.parse_args()

    if args.mode == "request":
        evaluate_with_openai(args.query_file, args.ans1_file,
                                args.ans2_file, args.output_file)
    elif args.mode == "result":
        results = read_eval_results(args.output_file)
        print_first_result(results)

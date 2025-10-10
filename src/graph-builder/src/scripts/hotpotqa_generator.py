import os
import re
import json

def sanitize_filename(title):
    """
    Removes characters that are invalid for file names across different OS
    and truncates the filename to a reasonable length.
    """
    if not title:
        return "Untitled"
    # Replace invalid characters with an underscore
    sanitized = re.sub(r'[<>:"/\\|?*\n]', '_', title)
    # Truncate to avoid "File name too long" errors
    return sanitized[:150]

def save_hotpotqa_json_to_txt(json_filepath, output_dir):
    """
    Saves the HotpotQA dataset in JSON format to text files.
    Each file is named after the question title, sanitized for file system compatibility.
    """
    # Load the dataset
    dataset = json.load(open(json_filepath, 'r', encoding='utf-8'))

    for item in dataset:
        title = item.get('title')
        sanitized_title = sanitize_filename(title)
        file_path = os.path.join(output_dir, f"{sanitized_title}.txt")

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(item.get('text'), f, ensure_ascii=False, indent=4)

def main():
    json_filepath = '/home/exouser/llm-knowledge-graph-builder/data/hotpotqa/corpus.json'  # Update with the actual path
    # Update with the desired output directory
    output_dir = '/home/exouser/llm-knowledge-graph-builder/data/hotpotqa/txt'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_hotpotqa_json_to_txt(json_filepath, output_dir)

if __name__ == "__main__":
    main()
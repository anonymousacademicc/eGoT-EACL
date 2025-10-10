import os
import re
from datasets import load_dataset
import pandas as pd

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

# 1. Load the dataset from Hugging Face
print("Loading dataset...")
ds = load_dataset("yixuantt/MultiHopRAG", "MultiHopRAG", split='train')
print("Dataset loaded.")

# 2. Filter the dataset for non-empty 'evidence_list'
print("Filtering for non-empty evidence lists...")
filtered_ds = ds.filter(lambda example: example['evidence_list'] is not None and len(example['evidence_list']) > 0)
print(f"Found {len(filtered_ds)} rows with evidence.")

# 3. Randomly select 1000 rows
print("Shuffling and selecting 1000 random rows...")
if len(filtered_ds) > 1000:
    sample_ds = filtered_ds.shuffle(seed=42).select(range(1000))
else:
    print(f"Warning: Only {len(filtered_ds)} valid rows found. Using all of them.")
    sample_ds = filtered_ds
print("Sample selected.")

# 4. Convert the selected dataset sample to a pandas DataFrame and save to CSV
print("Converting to DataFrame and saving to CSV...")
df_sample = sample_ds.to_pandas()
csv_filename = "/home/exouser/llm-knowledge-graph-builder/data/multihop-rag/selected_1000_rows.csv"
df_sample.to_csv(csv_filename, index=False)
print(f"Successfully saved the 1000 selected rows to '{csv_filename}'.")

# 5. Create a directory to store the individual evidence text files
output_dir = "/home/exouser/llm-knowledge-graph-builder/data/multihop-rag/contexts"
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory '{output_dir}' is ready.")

# 6. Iterate through the selected rows and create individual text files
file_counter = 0
print("Processing rows and creating individual files...")
for row in sample_ds:
    for evidence in row['evidence_list']:
        # Extract details, providing defaults if a key is missing
        title = evidence.get('title', 'No Title Provided')
        author = evidence.get('author', 'N/A')
        category = evidence.get('category', 'N/A')
        fact = evidence.get('fact', 'No Fact Provided')
        url = evidence.get('url', 'No URL Provided')

        # Create a safe filename from the title
        filename = sanitize_filename(title) + ".txt"
        filepath = os.path.join(output_dir, filename)

        # Format the content
        content = (
            f"Title: {title}\n"
            f"Author: {author}\n"
            f"Category: {category}\n\n"
            f"Fact: {fact}\n\n"
            f"URL: {url}\n"
        )

        # Write the content to the file
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            file_counter += 1
        except IOError as e:
            print(f"Error writing to file {filepath}: {e}")

print(f"\n✅ Done! Successfully created '{csv_filename}' and {file_counter} text files in the '{output_dir}' directory.")
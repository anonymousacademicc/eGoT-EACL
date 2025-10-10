# Dataset settings
DATASET = "mix"
MAX_QUERIES_MAP = {
    "mix": 130,
    "cs": 100,
    "agriculture": 100,
    "legal": 100,
    "sclc": 21,
}

MAX_QUERIES = MAX_QUERIES_MAP.get(DATASET, 100)

# Prompt constants
SYSTEM_PROMPT = """
---Role---
You are an expert tasked with evaluating two answers to the same question 
based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.
"""

USER_PROMPT_TEMPLATE = """
You will evaluate two answers to the same question based on three criteria: 
**Comprehensiveness**, **Diversity**, and **Empowerment**.

- **Comprehensiveness**: How much detail does the answer provide to cover all aspects and details of the question?  
- **Diversity**: How varied and rich is the answer in providing different perspectives and insights on the question?  
- **Empowerment**: How well does the answer help the reader understand and make informed judgments about the topic?  

For each criterion, choose the better answer (either Answer 1 or Answer 2) and explain why.  
Then, select an overall winner based on these three categories.  

Here is the question:
{query}

Here are the two answers:

**Answer 1:**
{answer1}

**Answer 2:**
{answer2}

Evaluate both answers using the three criteria listed above and provide detailed explanations.  
Be unbiased and fair.  

Output in JSON format:
{{
    "Comprehensiveness": {{
        "Winner": "[Answer 1 or Answer 2]",
        "Explanation": "[Explanation here]"
    }},
    "Empowerment": {{
        "Winner": "[Answer 1 or Answer 2]",
        "Explanation": "[Explanation here]"
    }},
    "Diversity": {{
        "Winner": "[Answer 1 or Answer 2]",
        "Explanation": "[Explanation here]"
    }},
    "Overall Winner": {{
        "Winner": "[Answer 1 or Answer 2]",
        "Explanation": "[Why this is the winner]"
    }}
}}
"""

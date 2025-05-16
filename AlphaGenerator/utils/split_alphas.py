import os
import random
import json
from google import genai
from google.genai import types

# Load environment variables
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]

client = genai.Client(api_key=GOOGLE_API_KEY)
# Neutralization category mapping
category_map = {
    "fundamental": ["industry"],
    "analyst": ["industry"],
    "model": ["industry", "subindustry", "sector", "market"],
    "news": ["subindustry"],
    "price-volume": ["market", "sector"],
    "social-media": ["industry", "subindustry"],
    "sentiment": ["industry", "subindustry"],
    "earnings": ["industry"],
    "macro": ["market", "sector", "industry"],
}


def truncate_contents(contents, max_tokens):
    truncated_contents = []
    token_count = 0

    for content in contents:
        content_tokens = len(content.split())
        if token_count + content_tokens <= max_tokens:
            truncated_contents.append(content)
            token_count += content_tokens
        else:
            remaining_tokens = max_tokens - token_count
            truncated_content = " ".join(content.split()[:remaining_tokens])
            truncated_contents.append(truncated_content)
            break

    return "\n".join(truncated_contents)


def generate_neutralization_prompt(alpha, max_tokens=50000):
    prompt = f"""
You are an AI assistant tasked with analyzing a financial alpha and the datasets used in it and providing ONLY and ONLY a JSON object of classifying the financial alpha into one of a few categories.

Your task is to thoroughly review the alpha given delineated by [ALPHA]:

[ALPHA]
{alpha[0]}
[ALPHA]

You need to categorize it into one of the following categories, in the format {{"category": "<category>"}}:
{list(category_map.keys())}

Please write the JSON object ONLY below:
"""
    return prompt.strip()


def split_alphas_neut(alphas):
    appended_alphas = []

    for alpha in alphas:
        prompt = generate_neutralization_prompt([alpha["code"]])

        try:
            res = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2,
                    max_output_tokens=1000
                )
            )

            json_text = res.text
            print(res.text)
            substring = json_text[json_text.find("{"): json_text.rfind("}") + 1]
            datasets = json.loads(substring, strict=False)

            print("Neutralizations classified")
            print(substring)

            alpha["neutralizations"] = category_map.get(datasets["category"], [])
            appended_alphas.append(alpha)

        except Exception as e:
            print(f"Error processing alpha: {e}")

    return appended_alphas


def split_alphas_region(alphas):
    new_alphas = []
    for alpha in alphas:
        alpha["region"] = "CHN"
        new_alphas.append(alpha)
    return new_alphas


def prune_datasets(alphas):
    new_alphas = []
    for alpha in alphas:
        relevant_datasets = []
        for data in alpha.get("datasets", []):
            if data["field"] in alpha.get("code", ""):
                relevant_datasets.append(data)
        alpha["datasets"] = relevant_datasets
        new_alphas.append(alpha)
    return new_alphas

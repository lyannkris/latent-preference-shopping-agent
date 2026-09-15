import json
import time
import os

from openai import OpenAI



INPUT_F = "data/items_shuffle_filtered_reduced.json"
OUTPUT_F = "curio_output/product_embeddings.json"

client = OpenAI(
    base_url="http://promaxgb10-d473.eecs.umich.edu:8000/v1",
    api_key="XX"
)

response = client.chat.completions.create(
model="openai/gpt-oss-120b",
messages=[
{"role": "user", "content": "Hello"}
]
)

print(response.choices[0].message.content)

MODEL_NAME = "openai/gpt-oss-120b" 


PROMPT_TEMPLATE = """
You are an expert product analyst.

Given the following product description, rate the product from 0 to 1 on these eight latent variables:

1. Comfort
2. Functionality
3. Maintenance
4. Status Symbol
5. Modesty
6. Sustainability
7. Risk Tolerance
8. Trend Sensitivity

Return ONLY a JSON dictionary with EXACTLY these keys:

[
  "comfort",
  "functionality",
  "maintenance",
  "status_symbol",
  "modesty",
  "sustainability",
  "risk_tolerance",
  "trend_sensitivity"
]

Each value must be a float between 0 and 1.

Product description:
{description}
"""


def embed_product(description):
    try:
        # Add description to the prompt template
        formatted_prompt = PROMPT_TEMPLATE.format(description=description)

        response = client.chat.completions.create(
            model=MODEL_NAME,
      
            messages=[{"role": "user", "content": formatted_prompt}],timeout=60.0
        )
        return json.loads(response.choices[0].message.content)
    
    except Exception as e:
        print(f"Error embedding product: {e}")
        time.sleep(5) 
        # avoid crashing on errors
        return {k: 0.5 for k in ["comfort", "functionality", "maintenance", "status_symbol", 
                                 "modesty", "sustainability", "risk_tolerance", "trend_sensitivity"]}


def main():
    if not os.path.exists("curio_output"):
        os.makedirs("curio_output")

    with open(INPUT_F, "r") as f:
        products = json.load(f)

    output = {}

    for idx, product in enumerate(products):
        a = product.get("asin", f"product_{idx}")
        description = (product.get("full_description") or product.get("name") or "")

        emb = embed_product(description)
        output[a] = emb

        time.sleep(0.1)

    with open(OUTPUT_F, "w") as f:
        json.dump(output, f, indent= 2)

    print(f"Finished.")


if __name__ == "__main__":
    main()
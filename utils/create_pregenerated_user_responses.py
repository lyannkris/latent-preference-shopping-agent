from generate_users import AGE_RANGE, GENDER_VALS, EVENT_VALS, SIZE_VALS, INCOME_VALS, LOCATION_VALS, CLIMATE_VALS, ACTIVITY_LEVEL_VALS, SOCIAL_FREQUENCY_VALS, WORLD_NEWS_AWARENESS_VALS, BRAND_AWARENESS_VALS, HIGHEST_EDUCATION_LEVEL_VALS, PERSONALITY_VALS, NAME_VALS, OCCUPATION_VALS, RELATIONSHIP_STATUS_VALS, LANGUAGES_SPOKEN_VALS, PLACES_TRAVELED_VALS, PET_OWNERSHIP_VALS, SIBLING_COUNT_RANGE, LIFE_GOALS_VALS
from generate_users import generate_by_LLM

import json
import random
import time
from typing import List

# --- CONFIGURATION ---
TOTAL_TARGET_RESPONSES = 100  
BATCH_SIZE = 10              
TRAIN_SPLIT = 0.8            

def fetch_responses_for_attribute(attr_name: str, attr_val: any) -> List[str]:
    """Fetches a large batch of diverse responses with intra-batch progress tracking."""
    accumulated_responses = []
    batches_needed = TOTAL_TARGET_RESPONSES // BATCH_SIZE

    for batch in range(batches_needed):
        batch_start_time = time.time()
        
        style_instruction = random.choice([
            "Keep the responses very short and literal (1-5 words).",
            "Make the responses highly colloquial, using slang and casual phrasing.",
            "Make the responses rambling, offering slightly too much context.",
            "Make the responses slightly evasive or sarcastic, though still answering.",
            "Include one response that dodges the question entirely (e.g., 'I don't see why that matters')."
        ])

        # --- OPTIMIZATION: Template Prompting for Age ---
        if attr_name == "age" and attr_val == "TEMPLATE":
            prompt = f"""You are generating diverse training data for a conversational AI. 
            You are asked: 'Could you tell me more about your age?'
            
            Generate exactly {BATCH_SIZE} distinct, natural conversational responses.
            CRITICAL: You must use the exact string "[AGE]" instead of a specific number. 
            Example: "I just turned [AGE] last month." or "I am [AGE] years old."
            
            STYLE CONSTRAINT: {style_instruction}
            CRITICAL: Output ONLY a valid JSON array of strings.
            """
        else:
            prompt = f"""You are generating diverse training data for a conversational AI. 
            Pretend to be a person whose '{attr_name}' is exactly '{attr_val}'.
            You are asked: 'Could you tell me more about your {attr_name}?'
            
            Generate exactly {BATCH_SIZE} distinct, natural conversational responses to this question.
            STYLE CONSTRAINT: {style_instruction}
            
            CRITICAL: Output ONLY a valid JSON array of strings. No markdown, no headers. 
            """
        
        existing_conversation = [{"role": "system", "content": prompt}]
        message = "Please begin generating the responses. Output only the JSON array."

        attempts = 0
        max_attempts = 3
        success = False
        
        while not success and attempts < max_attempts:
            attempts += 1
            reply, existing_conversation = generate_by_LLM(new_message=message, existing_conversation=existing_conversation)
            
            try:
                clean_reply = reply.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
                parsed_list = json.loads(clean_reply)
                
                if isinstance(parsed_list, list):
                    accumulated_responses.extend(parsed_list)
                    success = True
                else:
                    raise ValueError("Output was JSON, but not a list.")
                    
            except (json.JSONDecodeError, ValueError):
                if attempts < max_attempts:
                    message = "Your last response was not a valid JSON array. Please try again and output ONLY a JSON array."
                else:
                    print(f"\n      [!] Batch failed. Raw reply: {reply[:100]}...")

        # Intra-batch reporting
        elapsed_batch = time.time() - batch_start_time
        print(f"\n      [Batch {batch+1}/{batches_needed}] completed in {elapsed_batch:.2f}s", end="")

    return accumulated_responses


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    
    # Notice "age" is now just a single TEMPLATE string
    all_attributes = {
        "age": ["TEMPLATE"], 
        "sibling count": list(range(SIBLING_COUNT_RANGE[0], SIBLING_COUNT_RANGE[1] + 1)),
        "gender": GENDER_VALS,
        "attending event": EVENT_VALS,
        "clothing size": SIZE_VALS,
        "income": INCOME_VALS,
        "location": LOCATION_VALS,
        "climate": CLIMATE_VALS,
        "activity level": ACTIVITY_LEVEL_VALS,
        "social frequency": SOCIAL_FREQUENCY_VALS,
        "world news awareness": WORLD_NEWS_AWARENESS_VALS,
        "brand awareness": BRAND_AWARENESS_VALS,
        "highest education level": HIGHEST_EDUCATION_LEVEL_VALS,
        "personality": PERSONALITY_VALS,
        "name": NAME_VALS,
        "occupation": OCCUPATION_VALS,
        "relationship status": RELATIONSHIP_STATUS_VALS,
        "languages spoken": LANGUAGES_SPOKEN_VALS,
        "places traveled": PLACES_TRAVELED_VALS,
        "pet ownership": PET_OWNERSHIP_VALS,
        "life goals": LIFE_GOALS_VALS
    }

    train_dict = {}
    eval_dict = {}

    total_categories = len(all_attributes)
    
    for i, (attr_name, attr_values) in enumerate(all_attributes.items(), 1):
        print(f"\n\n--- Processing Category {i}/{total_categories}: '{attr_name}' ---")
        
        train_dict[attr_name] = {}
        eval_dict[attr_name] = {}
        
        for val in attr_values:
            print(f"\n  -> Value: '{val}'...", end="", flush=True)
            
            value_start_time = time.time()
            responses = fetch_responses_for_attribute(attr_name, val)
            
            responses = list(set(responses)) 
            random.shuffle(responses)
            
            split_idx = int(len(responses) * TRAIN_SPLIT)
            train_responses = responses[:split_idx]
            eval_responses = responses[split_idx:]
            
            # --- PROCEDURAL EXPANSION FOR AGE ---
            if attr_name == "age" and val == "TEMPLATE":
                print("\n  -> Expanding age templates from 15 to 65...", end="")
                for actual_age in range(AGE_RANGE[0], AGE_RANGE[1] + 1):
                    # Replace the placeholder with the actual integer for every string in the list
                    train_dict["age"][actual_age] = [r.replace("[AGE]", str(actual_age)) for r in train_responses]
                    eval_dict["age"][actual_age] = [r.replace("[AGE]", str(actual_age)) for r in eval_responses]
            else:
                train_dict[attr_name][val] = train_responses
                eval_dict[attr_name][val] = eval_responses
            
            value_elapsed = time.time() - value_start_time
            print(f"\n  -> Done! Total time: {value_elapsed:.2f}s (Yielded {len(responses)} unique responses)")

    print("\nGeneration Complete! Saving to disk...")
    
    with open("data/train_pregenerated_responses.json", "w") as f:
        json.dump(train_dict, f, indent=2)
        
    with open("data/eval_pregenerated_responses.json", "w") as f:
        json.dump(eval_dict, f, indent=2)

    print("Successfully saved splits to 'data/'")
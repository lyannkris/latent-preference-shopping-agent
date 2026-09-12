import ijson
import json
import random
from decimal import Decimal

def convert_decimal(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError

original_file = "data/items_shuffle_1000_filtered.json"
input_file = "data/items_shuffle_filtered.json"
output_file = "data/items_shuffle_filtered_reduced.json"

# Optimization: Using a set for original_asins makes the 'in' lookup significantly faster
original_asins = set()
filtered_items = []

print("Reading original file...")
with open(original_file, "r", encoding="utf-8") as infile:
    for obj in ijson.items(infile, "item"):
        original_asins.add(obj.get("asin"))
        filtered_items.append(obj)

# Calculate how many random items we still need to reach 1000
target_count = 10000
needed_items = target_count - len(filtered_items)

print(f"Original items loaded: {len(filtered_items)}. Need {needed_items} more items.")

if needed_items > 0:
    reservoir = []
    items_seen = 0
    
    print("Sampling from input file...")
    with open(input_file, "r", encoding="utf-8") as infile:
        for obj in ijson.items(infile, "item"):
            if obj.get("asin") in original_asins:
                continue
            
            # --- Reservoir Sampling Logic ---
            # Fill the reservoir until we have the needed amount
            if items_seen < needed_items:
                reservoir.append(obj)
            else:
                # Once full, randomly replace items with a decreasing probability
                j = random.randint(0, items_seen)
                if j < needed_items:
                    reservoir[j] = obj
                    
            items_seen += 1
            
            if items_seen % 100000 == 0:
                print(f"{items_seen} valid items scanned through")

    # Add the randomly selected items to our final list
    filtered_items.extend(reservoir)
    
elif needed_items < 0:
    # If the original file somehow already had more than 1000 items, just truncate it
    print("Original file already has over 1000 items. Truncating to 1000.")
    filtered_items = filtered_items[:target_count]

# Optional: Shuffle the final combined list so the old and new items are mixed
random.shuffle(filtered_items)

print(f"Final count: {len(filtered_items)}. Creating output json...")
with open(output_file, "w", encoding="utf-8") as outfile:
    json.dump(filtered_items, outfile, default=convert_decimal)
import ijson
import json
from decimal import Decimal

def convert_decimal(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError

clothing_tag = "Clothing, Shoes & Jewelry"
filtered_asins = []

# different jsons have different formats, so going to make the format consistent with the original file format for now
# this is the large 5gb file, uses [] to enclose the json
input_file =  "data/items_shuffle.json"
output_file = "data/items_shuffle_filtered.json"
print(f"filtering {input_file}")

filtered_items = []

count = 0
with open(input_file, "r", encoding="utf-8") as infile:
    for obj in ijson.items(infile, "item"):
        if clothing_tag in obj.get("product_category"):
            filtered_asins.append(obj.get("asin"))
            filtered_items.append(obj)
        count += 1
        if count % 100000 == 0:
            print(f"{count} items scanned through")

print("creating output json")
with open(output_file, "w", encoding="utf-8") as outfile:
    json.dump(filtered_items, outfile, default=convert_decimal)


filtered_asins = set(filtered_asins)

# this is the small version of the large file above, uses [] to enclose the json
input_file =  "data/items_shuffle_1000.json"
output_file = "data/items_shuffle_1000_filtered.json"
print(f"filtering {input_file}")

filtered_items = []
count = 0
with open(input_file, "r", encoding="utf-8") as infile:
    for obj in ijson.items(infile, "item"):
        if clothing_tag in obj.get("product_category"):
            filtered_items.append(obj)
        count += 1
        if count % 100000 == 0:
            print(f"{count} items scanned through")

print("creating output json")
with open(output_file, "w", encoding="utf-8") as outfile:
    json.dump(filtered_items, outfile, default=convert_decimal)

# this uses {}
input_file =  "data/items_human_ins.json"
output_file = "data/items_human_ins_filtered.json"
print(f"filtering {input_file}")

filtered_items = {}
count = 0
with open(input_file, "r", encoding="utf-8") as infile:
    for asin, data in ijson.kvitems(infile, ""):
        if asin in filtered_asins:
            filtered_items[asin] = data
        count += 1
        if count % 100000 == 0:
            print(f"{count} items scanned through")

print("creating output json")
with open(output_file, "w", encoding="utf-8") as outfile:
    json.dump(filtered_items, outfile, default=convert_decimal)

# this uses {}
input_file =  "data/items_ins_v2_1000.json"
output_file = "data/items_ins_v2_1000_filtered.json"
print(f"filtering {input_file}")

filtered_items = {}
count = 0
with open(input_file, "r", encoding="utf-8") as infile:
    for asin, data in ijson.kvitems(infile, ""):
        if asin in filtered_asins:
            filtered_items[asin] = data
        count += 1
        if count % 100000 == 0:
            print(f"{count} items scanned through")

print("creating output json")
with open(output_file, "w", encoding="utf-8") as outfile:
    json.dump(filtered_items, outfile, default=convert_decimal)

# this uses {}
input_file =  "data/items_ins_v2.json"
output_file = "data/items_ins_v2_filtered.json"
print(f"filtering {input_file}")

filtered_items = {}
count = 0
with open(input_file, "r", encoding="utf-8") as infile:
    for asin, data in ijson.kvitems(infile, ""):
        if asin in filtered_asins:
            filtered_items[asin] = data
        count += 1
        if count % 100000 == 0:
            print(f"{count} items scanned through")

print("creating output json")
with open(output_file, "w", encoding="utf-8") as outfile:
    json.dump(filtered_items, outfile, default=convert_decimal)
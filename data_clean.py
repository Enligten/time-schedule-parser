import json
import re

with open("train_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

def remove_number_prefix(text):
    return re.sub(r'^\d+\.\s*', '', text.strip())

# Batch processing of data
for item in data:
    item["sentence"] = remove_number_prefix(item["sentence"])

# Check the processing results
print(data[0]["sentence"]) 

# Save the processed data
with open("train_data_cleaned.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)


from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM
from datasets import load_dataset

dataset = load_dataset("json", data_files="train_data_cleaned.json")["train"]
dataset = dataset.shuffle(seed=42)
# Split the dataset
train_test_split = dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

print("dataset origin1: ",train_dataset[0])
print("dataset origin2: ",test_dataset[0])

model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def preprocess_function(examples):
    inputs = ["extract time, location, and event: " + text for text in examples["sentence"]]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    # print("inputs tokenizer: ", model_inputs[1])
    clean_labels = [label.strip() for label in examples["label"]]
    labels = tokenizer(clean_labels, max_length=128, truncation=True, padding="max_length")
    print("lables tokenizer: ", clean_labels[1])
    model_inputs["labels"] = labels["input_ids"]
    # print("model_inputs tokenizer: ", model_inputs["labels"][1])
    return {
        "input_ids": model_inputs["input_ids"],
        "attention_mask": model_inputs["attention_mask"],
        "labels": labels["input_ids"]
    }

# Process the entire dataset
train_tokenized_datasets = train_dataset.map(preprocess_function, batched=True, remove_columns=["sentence", "label"])
test_tokenized_datasets = test_dataset.map(preprocess_function, batched=True, remove_columns=["sentence", "label"])
# data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)
print(train_tokenized_datasets[0])
print(test_tokenized_datasets[0])
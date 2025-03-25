from transformers import AutoModelForSeq2SeqLM, TrainingArguments, Trainer, AutoTokenizer
from process_data import train_tokenized_datasets, test_tokenized_datasets
import torch

# Load the model and tokenizer
model_name = "t5-small"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device) 
# Define the training arguments
training_args = TrainingArguments(
  output_dir="./results", 
  num_train_epochs=3,
  learning_rate=2e-5,
  per_device_train_batch_size=8,
  per_device_eval_batch_size=8,
  evaluation_strategy="epoch",
  save_strategy="epoch",
  save_total_limit=3,
  weight_decay=0.01,
  fp16 = True,
)

# Define the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized_datasets,
    eval_dataset=test_tokenized_datasets,
    tokenizer=tokenizer,
)

#Check if the label length is aligned
for i in range(min(5, len(train_tokenized_datasets))):
    print(f"Sample {i} labels length: {len(train_tokenized_datasets[i]['labels'])}")

# Train the model
trainer.train()

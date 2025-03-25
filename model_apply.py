from transformers import AutoTokenizer, T5ForConditionalGeneration

model_path = r"results\checkpoint-1227"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

def summarize(text):
    inputs = tokenizer("extract time, location, and event: " + text, return_tensors="pt", max_length=128, truncation=True, padding=True)
    outputs = model.generate(**inputs)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

def main():
    test_text = input("Please enter your schedule: ")
    print("********************************")
    print("time|location|event")
    print(summarize(test_text))
    print("********************************")

if __name__ == "__main__":
    main()
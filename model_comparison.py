
from transformers import AutoTokenizer, T5ForConditionalGeneration

model_path = r"results\checkpoint-1227"
model_path2 = "t5-small"
tokenizer2 = AutoTokenizer.from_pretrained(model_path2)
model2 = T5ForConditionalGeneration.from_pretrained(model_path2)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

def summarize(text):
    inputs = tokenizer("extract time, location, and event: " + text, return_tensors="pt", max_length=128, truncation=True, padding=True)
    # print("Input tokens:", inputs)
    outputs = model.generate(**inputs)
    #print("Generated token ids:", outputs)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

def summarize2(text):
    inputs = tokenizer2("extract time, location, and event: " + text, return_tensors="pt", max_length=128, truncation=True, padding=True)
    # print("Input tokens:", inputs)
    outputs = model2.generate(**inputs)
    #print("Generated token ids:", outputs)
    summary = tokenizer2.decode(outputs[0], skip_special_tokens=True)
    return summary

# test
test_text = "I'm going to Lhasa on Saturday morning"
print("Generated Summary:", summarize(test_text))
print("Generated Summary:", summarize2(test_text))
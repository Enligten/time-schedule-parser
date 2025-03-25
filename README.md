# time-schedule-parser
This is a small Domo made for the "AI for Science" project in GMoc.

A fine-tuned T5 model for extracting time, location and event from text.
The fine-tuning model is too big to upload to github. So the result file is incomplete and you can't call it directly.

File function introduction
1. generate_data.py: generate 4000 training data and save them into train_data.json file.
2. data_clean.py: performed data cleaning, and formatted them for structured learning, and save them into train_data_cleaned.json file.
3. process_data.py: the training data is tokenized and padded to the same length.
4. fine_tuning.py: Fine-tuning the T5-small model.
5. model_comparison.py: compare the output differences between the fine-tuned model and the original model.
6. model_apply.py: in this file, you can experiment with fine-tuned model.

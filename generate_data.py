# pip install openai-agents
import openai
import json
import time

# Set up your OpenAI API key
openai.api_key = " "  # Replace with the actual API key

# Define the prompt to send to ChatGPT
system_message = {
    "role": "system",
    "content": """
You need to generate training data. You need to write sentences about schedule in which include time, location and main event, and lables including time, location and event. 
Your outputs must follow this template and write to CSV format:(Sentence), (time), (location), (event)
Do not mark the serial number and Do not say any other sentence.
There is an output examples:
I have a math class tomorrow afternoon at 3:00, tomorrow 3:00 PM, NONE, math class
There is an online meeting on March 6th, March 6th, online, meeting
I'm going to Lhasa on Saturday, Saturday, Lhasa, NONE
"""}

user_message = {
    "role": "user",
    "content": "Please generate 100 new sentences."
}

# Initialize an empty list to store the generated data
data_entries = []

# Defines the number of data entries generated per request
entries_per_request = 100

# Define the total number of data entries that need to be generated
total_entries = 4000

# Calculate the number of requests required
num_requests = total_entries // entries_per_request
if total_entries % entries_per_request != 0:
    num_requests += 1

#Loop to send requests to generate data
for i in range(num_requests):
    try:
        # Call ChatGPT API
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[system_message, user_message],
            temperature=0.5, 
        )
        # Parsing the response content
        response_text = response.choices[0].message.content
        #print("Response Text:", response_text)
        count = 0
        for line in response_text.split("\n"):
            count += 1
            if line != "": #avoid empty line
            # Split the generated data into sentences and labels
                sentence, times, location, event = line.split(",")
                # Add data to a list
                data_entries.append({
                    "sentence": sentence,
                    "label": times + "," + location + "," + event
                })
        # Print progress
        print(f"generate {count} data items, total: {len(data_entries)}")
        time.sleep(5)
    except Exception as e:
        print(f"request {i+1} error:{e}")

# Writing data to a JSON file
with open("train_data.json", "w", encoding="utf-8") as f:
    json.dump(data_entries, f, indent=4, ensure_ascii=False)

print("Data generation completed!")
# IBM - Building Generative AI-Powered Applications with Python
IBM - Building Generative AI-Powered Applications with Python

* DeepLearning.AI TensorFlow Developer [DeepLearning.AI]( https://github.com/jkaewprateep/Portfolio/blob/main/Coursera%20HRU25ZPYYDK5.pdf )
* IBM AI Engineering [IBM]( https://github.com/jkaewprateep/Portfolio/blob/main/Coursera%20SX6XYWRRRNQZ.pdf )
* IBM AI Developer Professional [IBM]( https://github.com/jkaewprateep/Portfolio/blob/main/Coursera%20PLRYP4UYK4T2.pdf )
* IBM AI Product Manager [IBM]( https://github.com/jkaewprateep/Portfolio/blob/main/Coursera%20QT6UPKZRFJMV.pdf )
* Applied Data Science with Python [University of Michigan]( https://github.com/jkaewprateep/Portfolio/blob/main/Coursera%20ZG9J39MKQAXC.pdf )

<p align="center" width="100%">
    <img width="47%" src="https://github.com/jkaewprateep/IBM---Building-Generative-AI-Powered-Applications-with-Python/blob/main/IBM%20-%20Building%20Generative%20AI-Powered%20Applications%20with%20Python%20-%20instructors.png">
    <img width="17.63%" src="https://github.com/jkaewprateep/IBM---Building-Generative-AI-Powered-Applications-with-Python/blob/main/kid_36.jpg">
    <img width="17.63%" src="https://github.com/jkaewprateep/IBM---Building-Generative-AI-Powered-Applications-with-Python/blob/main/kid_37.jpg"> </br>
    <b> Pictures from the Internet </b> </br>
</p>

## Application ChatBot ##

🧸💬 There are a variety of AI applications but in the middle of development understanding and response is hiring AI to perform repeat programmable tasks, expandable, searching and organizing, and concentration tasks. </br>

<p align="center" width="100%">
    <img width="60%" src="https://github.com/jkaewprateep/IBM---Building-Generative-AI-Powered-Applications-with-Python/blob/main/Screenshot%202024-09-03%20143009.png"> </br>
    <b> Pictures from the Internet </b> </br>
</p>

### How to train AI for continuous learning ###

🧸💬 Training AI with data augmentation known method for many AI machine learning developers, why a cat they understand of the mirror and random mirrors with the same information⁉️ That is because they are learnable. </br>

<p align="center" width="100%">
    <img width="40%" src="https://github.com/jkaewprateep/IBM---Building-Generative-AI-Powered-Applications-with-Python/blob/main/cat_06.jpg"> </br>
    <b> Pictures from the Internet </b> </br>
</p>

#### Find the fastest way or rewards return ####

<p align="center" width="100%">
    <img width="40%" src="https://github.com/jkaewprateep/IBM---Building-Generative-AI-Powered-Applications-with-Python/blob/main/cat_07.jpg"> </br>
    <b> Pictures from the Internet </b> </br>
</p>

#### New challenge continues ####

<p align="center" width="100%">
    <img width="40%" src="https://github.com/jkaewprateep/IBM---Building-Generative-AI-Powered-Applications-with-Python/blob/main/cat_08.jpg"> </br>
    <b> Pictures from the Internet </b> </br>
</p>

#### Concentration ####

<p align="center" width="100%">
    <img width="40%" src="https://github.com/jkaewprateep/IBM---Building-Generative-AI-Powered-Applications-with-Python/blob/main/cat_09.jpg"> </br>
    <b> Pictures from the Internet </b> </br>
</p>

#### Try best for rewards ####

<p align="center" width="100%">
    <img width="40%" src="https://github.com/jkaewprateep/IBM---Building-Generative-AI-Powered-Applications-with-Python/blob/main/cat_10.jpg"> </br>
    <b> Pictures from the Internet </b> </br>
</p>

### Implementation ###

#### server.py ####

```
import base64
import json
from flask import Flask, render_template, request
from worker import speech_to_text, text_to_speech, openai_process_message
from flask_cors import CORS
import os

# Add
from worker import speech_to_text, text_to_speech, openai_process_message

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/speech-to-text', methods=['POST'])
def speech_to_text_route():
    print("processing speech-to-text")
    audio_binary = request.data # Get the user's speech from their request
    text = speech_to_text(audio_binary) # Call speech_to_text function to transcribe the speech

    # Return the response back to the user in JSON format
    response = app.response_class(
        response=json.dumps({'text': text}),
        status=200,
        mimetype='application/json'
    )
    print(response)
    print(response.data)
    return response


@app.route('/process-message', methods=['POST'])
def process_message_route():
    user_message = request.json['userMessage'] # Get user's message from their request
    print('user_message', user_message)

    voice = request.json['voice'] # Get user's preferred voice from their request
    print('voice', voice)

    # Call openai_process_message function to process the user's message and get a response back
    openai_response_text = openai_process_message(user_message)

    # Clean the response to remove any emptylines
    openai_response_text = os.linesep.join([s for s in openai_response_text.splitlines() if s])

    # Call our text_to_speech function to convert OpenAI Api's reponse to speech
    openai_response_speech = text_to_speech(openai_response_text, voice)

    # convert openai_response_speech to base64 string so it can be sent back in the JSON response
    openai_response_speech = base64.b64encode(openai_response_speech).decode('utf-8')

    # Send a JSON response back to the user containing their message's response both in text and speech formats
    response = app.response_class(
        response=json.dumps({"openaiResponseText": openai_response_text, "openaiResponseSpeech": openai_response_speech}),
        status=200,
        mimetype='application/json'
    )

    print(response)
    return response


if __name__ == "__main__":
    app.run(port=8000, host='0.0.0.0')
```

### transformers.py ###

```
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Step 3: Choosing a model
model_name = "meta-llama/Meta-Llama-Guard-2-8B";
# model_name = "facebook/blenderbot-400M-distill"

# Step 4: Fetch the model and initialize a tokenizer
# Load model (download on first run and reference local installation for consequent runs)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name);
tokenizer = AutoTokenizer.from_pretrained(model_name);

# Step 5.1: Keeping track of conversation history
conversation_history = [];

# Step 5.2: Encoding the conversation history
history_string = "\n".join(conversation_history);

# Step 5.3: Fetch prompt from user
input_text ="hello, how are you doing?"

# Step 5.4: Tokenization of user prompt and chat history
inputs = tokenizer.encode_plus(history_string, input_text, return_tensors="pt")
print(inputs)

# Step 5.5: Generate output from the model
outputs = model.generate(**inputs)
print(outputs)

# Step 5.6: Decode output
response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
print(response)

# Step 5.7: Update conversation history
conversation_history.append(input_text)
conversation_history.append(response)
print(conversation_history)

# Step 6: Repeat
while True:
    # Create conversation history string
    history_string = "\n".join(conversation_history)

    # Get the input data from the user
    input_text = input("> ")

    # Tokenize the input text and history
    inputs = tokenizer.encode_plus(history_string, input_text, return_tensors="pt")

    # Generate the response from the model
    outputs = model.generate(**inputs)

    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    
    print(response)

    # Add interaction to conversation history
    conversation_history.append(input_text)
    conversation_history.append(response)
```

### app.py ###

```
from flask import Flask, render_template            # newly added
from flask_cors import CORS                         # newly added

from transformers import AutoModelForSeq2SeqLM      # newly added
from transformers import AutoTokenizer              # newly added

from flask import request                           # newly added
import json                                         # newly added

"""""""""""""""""""""""""""""""""""""""""""""""""""""
MODEL DEFINED
"""""""""""""""""""""""""""""""""""""""""""""""""""""
model_name = "facebook/blenderbot-400M-distill"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
conversation_history = []

"""""""""""""""""""""""""""""""""""""""""""""""""""""
EXPECTED MESSAGE
"""""""""""""""""""""""""""""""""""""""""""""""""""""
expected_message = {
    'prompt': 'message'
}

app = Flask(__name__)
CORS(app);                                          # newly added

# @app.route('/')
# def home():
#     return '🧸💬 Hello, World!'

@app.route('/bananas')
def bananas():
    return '🍌 This page has bananas!'
    
@app.route('/bread')
def bread():
    return '🍞 This page has bread!'

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/chatbot', methods=['POST'])
def handle_prompt():
    # Read prompt from HTTP request body
    data = request.get_data(as_text=True)
    data = json.loads(data)
    input_text = data['prompt']

    # Create conversation history string
    history = "\n".join(conversation_history)

    # Tokenize the input text and history
    inputs = tokenizer.encode_plus(history, input_text, return_tensors="pt")

    # Generate the response from the model
    outputs = model.generate(**inputs, max_length= 60)  # max_length will acuse model to crash at some point as history grows

    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Add interaction to conversation history
    conversation_history.append(input_text)
    conversation_history.append(response)

    return response

if __name__ == '__main__':
    app.run()
```

### worker.py ###

```
def text_to_speech(text, voice=""):
    # Set up Watson Text-to-Speech HTTP Api url
    base_url = "https://sn-watson-stt.labs.skills.network";
    api_url = base_url + '/text-to-speech/api/v1/synthesize?output=output_text.wav'

    # Adding voice parameter in api_url if the user has selected a preferred voice
    if voice != "" and voice != "default":
        api_url += "&voice=" + voice

    # Set the headers for our HTTP request
    headers = {
        'Accept': 'audio/wav',
        'Content-Type': 'application/json',
    }

    # Set the body of our HTTP request
    json_data = {
        'text': text,
    }

    # Send a HTTP Post request to Watson Text-to-Speech Service
    response = requests.post(api_url, headers=headers, json=json_data)
    print('text to speech response:', response)
    return response.content
```

## Configuration ##

### watson.ai ###

```
https://jkaewprateep-8000.theiadockernext-1-labs-prod-theiak8s-4-tor01.proxy.cognitiveclass.ai/speech-to-text/api/v1

curl -X POST -H "Content-Type: application/json" -d '{"prompt": "Hello, how are you today?"}'
	https://jkaewprateep-8000.theiadockernext-1-labs-prod-theiak8s-4-tor01.proxy.cognitiveclass.ai/
	text-to-speech/api/v1/synthesize?output=output_text.wav

curl -X POST -H "Content-Type: application/json" -d '{"prompt": "Hello, how are you today?"}'
	https://jkaewprateep-8000.theiadockernext-1-labs-prod-theiak8s-4-tor01.proxy.cognitiveclass.ai/process-message

curl "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29" \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json' \
  -H 'Authorization: Bearer <long encryption authentication credential>' \
  -d '{
	"input": "",
	"parameters": {
		"decoding_method": "greedy",
		"max_new_tokens": 200,
		"min_new_tokens": 0,
		"stop_sequences": [],
		"repetition_penalty": 1
	},
	"model_id": "ibm/granite-13b-chat-v2",
	"project_id": "c276757e-855e-413d-868c-c1f3b312c8ce",
	"moderations": {
		"hap": {
			"input": {
				"enabled": true,
				"threshold": 0.5,
				"mask": {
					"remove_entity_value": true
				}
			},
			"output": {
				"enabled": true,
				"threshold": 0.5,
				"mask": {
					"remove_entity_value": true
				}
			}
		}
	}
}'

-----------------------
# Generate an IAM token by using an API key
curl -X POST 'https://iam.cloud.ibm.com/identity/token' -H 'Content-Type: application/x-www-form-urlencoded'
	-d 'grant_type=urn:ibm:params:oauth:grant-type:apikey&apikey=qArTNzrr9cC42N7I6D-lt_t9KylxtDVtwKvu6FvoHyWx'


docker build . -t voice-translator-powered-by-watsonx
docker run -p 8001:8001 voice-translator-powered-by-watsonx
```

### hugging face-cli ###

```
huggingface-cli login
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --include "original/*" --local-dir Meta-Llama-3-8B-Instruct
```

### Sample ###

<p align="center" width="100%">
	<img width="25%" src="https://github.com/jkaewprateep/IBM---Building-Generative-AI-Powered-Applications-with-Python/blob/main/web01.png">
	<img width="25%" src="https://github.com/jkaewprateep/IBM---Building-Generative-AI-Powered-Applications-with-Python/blob/main/web02.png">
	<img width="25%" src="https://github.com/jkaewprateep/IBM---Building-Generative-AI-Powered-Applications-with-Python/blob/main/web03.png">
	<img width="25%" src="https://github.com/jkaewprateep/IBM---Building-Generative-AI-Powered-Applications-with-Python/blob/main/web04.png">
	<img width="25%" src="https://github.com/jkaewprateep/IBM---Building-Generative-AI-Powered-Applications-with-Python/blob/main/web05.png">
	<img width="25%" src="https://github.com/jkaewprateep/IBM---Building-Generative-AI-Powered-Applications-with-Python/blob/main/web06.png">
	<img width="25%" src="https://github.com/jkaewprateep/IBM---Building-Generative-AI-Powered-Applications-with-Python/blob/main/web07.png">
	<img width="25%" src="https://github.com/jkaewprateep/IBM---Building-Generative-AI-Powered-Applications-with-Python/blob/main/web08.png">
	<img width="25%" src="https://github.com/jkaewprateep/IBM---Building-Generative-AI-Powered-Applications-with-Python/blob/main/web09.png">
	<img width="25%" src="https://github.com/jkaewprateep/IBM---Building-Generative-AI-Powered-Applications-with-Python/blob/main/web10.png">
	<img width="25%" src="https://github.com/jkaewprateep/IBM---Building-Generative-AI-Powered-Applications-with-Python/blob/main/web11.png">
	<img width="25%" src="https://github.com/jkaewprateep/IBM---Building-Generative-AI-Powered-Applications-with-Python/blob/main/web12.png">
	<img width="25%" src="https://github.com/jkaewprateep/IBM---Building-Generative-AI-Powered-Applications-with-Python/blob/main/web13.png">
	<img width="25%" src="https://github.com/jkaewprateep/IBM---Building-Generative-AI-Powered-Applications-with-Python/blob/main/web14.png">
	<img width="25%" src="https://github.com/jkaewprateep/IBM---Building-Generative-AI-Powered-Applications-with-Python/blob/main/web15.png">
	<img width="25%" src="https://github.com/jkaewprateep/IBM---Building-Generative-AI-Powered-Applications-with-Python/blob/main/web16.png">
	<img width="25%" src="https://github.com/jkaewprateep/IBM---Building-Generative-AI-Powered-Applications-with-Python/blob/main/web17.png">
	<img width="25%" src="https://github.com/jkaewprateep/IBM---Building-Generative-AI-Powered-Applications-with-Python/blob/main/web18.png">
	<img width="25%" src="https://github.com/jkaewprateep/IBM---Building-Generative-AI-Powered-Applications-with-Python/blob/main/web19.png">
	<img width="25%" src="https://github.com/jkaewprateep/IBM---Building-Generative-AI-Powered-Applications-with-Python/blob/main/web20.png">
	<img width="25%" src="https://github.com/jkaewprateep/IBM---Building-Generative-AI-Powered-Applications-with-Python/blob/main/web21.png">
	<img width="25%" src="https://github.com/jkaewprateep/IBM---Building-Generative-AI-Powered-Applications-with-Python/blob/main/web22.png">
	<img width="25%" src="https://github.com/jkaewprateep/IBM---Building-Generative-AI-Powered-Applications-with-Python/blob/main/web23.png">
	<img width="25%" src="https://github.com/jkaewprateep/IBM---Building-Generative-AI-Powered-Applications-with-Python/blob/main/web24.png">
	<img width="25%" src="https://github.com/jkaewprateep/IBM---Building-Generative-AI-Powered-Applications-with-Python/blob/main/web25.png">
</p>

---

<p align="center" width="100%">
    <img width="30%" src="https://github.com/jkaewprateep/advanced_mysql_topics_notes/blob/main/custom_dataset.png">
    <img width="30%" src="https://github.com/jkaewprateep/advanced_mysql_topics_notes/blob/main/custom_dataset_2.png"> </br>
    <b> 🥺💬 รับจ้างเขียน functions </b> </br>
</p>

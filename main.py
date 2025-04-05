from transformers import pipeline
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

# using flask to host the api, a front end will be created eventually
# check terminal for port, should be 5000 - if different, edit sample request
# sample request (copy and paste into terminal):  curl -X POST -H "Content-Type: application/json" -d '{"text": "I hate you"}' http://localhost:5000/sentiment-analysis
app = Flask(__name__)
CORS(app)

@app.route("/sentiment-analysis", methods=['POST'])
def hello():
    data = request.get_json()
    text = data['text']
    # pipeline is from huggingface can access pretrained models
    # https://huggingface.co/docs/transformers/main/en/main_classes/pipelines
    classifier = pipeline("sentiment-analysis")
    res = classifier(text)
    return jsonify({"classifier": "sentiment-analysis", "result": res})

if __name__ == '__main__':
    app.run(debug=True)

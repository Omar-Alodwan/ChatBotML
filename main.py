import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
annotations = json.loads(open('intents.json').read())
words = pickle.load(open(f'words.pkl','rb'))
classes = pickle.load(open(f'classes.pkl','rb'))
model = load_model(f'chatbot_model.h5')




def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    p = bag_of_words(sentence, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(annotations_list, annotations_json):
    try:
        type = annotations_list[0]['intent']
        list_of_annotations = annotations_json['annotations']
        for i in list_of_annotations:
            if i['type'] == type:
                result = random.choice(i['answer'])
                break
    except IndexError:
        result = "I don't understand!"
    return result

print("go ogogo")
while True:
    message = input("")
    ints = predict_class(message)
    res = get_response(ints, annotations)
    print(res)
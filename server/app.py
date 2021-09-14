from flask import Flask, json, request, render_template, jsonify
from flask.helpers import send_file
from werkzeug.exceptions import Forbidden, HTTPException, NotFound, RequestTimeout, Unauthorized
from werkzeug.utils import secure_filename
import os
import pandas as pd

from kafka import KafkaProducer
from kafka import KafkaConsumer
import codecs

app = Flask(__name__)

producer = KafkaProducer(bootstrap_servers=["b-1.demo-cluster-1.9q7lp7.c1.kafka.eu-west-1.amazonaws.com:9092",
    "b-2.demo-cluster-1.9q7lp7.c1.kafka.eu-west-1.amazonaws.com:9092"],api_version = (0,10,1))


consumer = KafkaConsumer('group7_text_topic',
                             client_id='d_id',
                             bootstrap_servers=["b-1.demo-cluster-1.9q7lp7.c1.kafka.eu-west-1.amazonaws.com:9092",
    "b-2.demo-cluster-1.9q7lp7.c1.kafka.eu-west-1.amazonaws.com:9092"],
                             auto_offset_reset='earliest',
                             enable_auto_commit = False
                             )



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/volunteer', methods = ['GET', 'POST'])
def get_audio():
    if request.method == 'GET':
        print("inside volunteer get request")
        last_msg = consumer.poll(timeout_ms=100,max_records=1)
        last_key = consumer.poll(timeout_ms=100,max_records=1)
        print("TYPE++++++++++", type(last_msg))
        print(last_msg)
        conv = list(last_msg.values())[0][0].value
        conv_key = list(last_key.values())[0][0].value
        print("TYPE++++++++++", type(conv))
        # text = conv.decode()
        conv = conv.decode()

        # print(text)
        # text = list(text)
        # text = text[0]
 
        key = "sentence 1"
        content = {
            "key":conv_key,
            "text":conv
        }
        return render_template('volunteer.html',content=content)



    if request.method == 'POST':
        # Get the file from post request
        # data = request.data
        print("+++===============================================")
        # data = request.form
        # blob = request.files

        
        blob = request.files['blob'].read()

        print("FILE===================",blob)

        print(request.form.getlist('fname'))

        producer.send("group7_text", bytes('audioname', encoding="utf-8"))
        producer.send("group7_audio",blob)
        return 'Done'
    


@app.errorhandler(NotFound)
def page_not_found_handler(e: HTTPException):
    return '<h1>404.html</h1>', 404


@app.errorhandler(Unauthorized)
def unauthorized_handler(e: HTTPException):
    return '<h1>401.html</h1>', 401


@app.errorhandler(Forbidden)
def forbidden_handler(e: HTTPException):
    return '<h1>403.html</h1>', 403


@app.errorhandler(RequestTimeout)
def request_timeout_handler(e: HTTPException):
    return '<h1>408.html</h1>', 408

if __name__ == '__main__':
    os.environ.setdefault('Flask_SETTINGS_MODULE', 'helloworld.settings')
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    port = int(os.environ.get("PORT", 6001))
    app.run(debug=True, port=9999)

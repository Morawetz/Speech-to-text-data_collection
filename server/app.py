from flask import Flask, request, render_template
from flask.helpers import send_file
from werkzeug.exceptions import Forbidden, HTTPException, NotFound, RequestTimeout, Unauthorized
from werkzeug.utils import secure_filename
import os
import pandas as pd

from kafka import KafkaProducer
from kafka import KafkaConsumer

app = Flask(__name__)

producer = KafkaProducer(bootstrap_servers=["b-1.demo-cluster-1.9q7lp7.c1.kafka.eu-west-1.amazonaws.com:9092",
    "b-2.demo-cluster-1.9q7lp7.c1.kafka.eu-west-1.amazonaws.com:9092"],api_version = (0,10,1))


consumer = KafkaConsumer('group6_test', client_id='d_id',
                             bootstrap_servers=["b-1.demo-cluster-1.9q7lp7.c1.kafka.eu-west-1.amazonaws.com:9092","b-2.demo-cluster-1.9q7lp7.c1.kafka.eu-west-1.amazonaws.com:9092"],
                             auto_offset_reset='earliest',
                             enable_auto_commit=False)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/volunteer', methods = ['GET', 'POST'])
def get_audio():
    if request.method == 'GET':
        # randomly select text and send to user
        text =  "አገራችን ከአፍሪካም ሆነ ከሌሎች የአለም አገራት ጋር ያላትን አለም አቀፋዊ ግንኙነት ወደ ላቀ ደረጃ ያሸጋገረ ሆኗል በአገር ውስጥ አራት አለም ጀልባያውም የወረቀት"
        return render_template('volunteer.html',data=text)
      
    if request.method == 'POST':
        # Get the file from post request
        f = request.data
        print(type(f))
        print(f)

        producer.send("group6_test",f)
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
    port = int(os.environ.get("PORT", 33507))
    app.run(debug=True)

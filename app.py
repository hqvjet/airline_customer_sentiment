from flask import Flask, request
from appService import getRatingFromModel

app = Flask(__name__)

@app.route('/', methods=['POST'])
def index():
    req = request.get_json()
    rating = getRatingFromModel(req.get('title'), req.get('text'))
    return rating

app.run(debug=False)
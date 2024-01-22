from flask import Flask, request, jsonify
from flask_cors import CORS
from appService import getRatingFromModel

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['POST'])
def index():
    req = request.get_json()
    rating = getRatingFromModel(req.get('title'), req.get('text'))
    data = rating.tolist()
    return jsonify({'data': data})

app.run(host='0.0.0.0', port=8081, debug=False)

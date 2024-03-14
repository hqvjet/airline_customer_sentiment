from flask import Flask, request, jsonify
from flask_cors import CORS

from glove import glove

app = Flask(__name__)
CORS(app)

# app.register_blueprint(airlines)
app.register_blueprint(glove)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8804, debug=False)

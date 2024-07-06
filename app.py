from flask import Flask, request, jsonify
from flask_cors import CORS

from phobert_bp import phobert

app = Flask(__name__)
CORS(app)

app.register_blueprint(phobert)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8804, debug=False)

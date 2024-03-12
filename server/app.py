from flask import Flask, request, jsonify
from flask_cors import CORS

from airline import airlines
from comment import comments

app = Flask(__name__)
CORS(app)

app.register_blueprint(airlines)
app.register_blueprint(comments)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=4000, debug=False)

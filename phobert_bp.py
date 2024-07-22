from flask import Blueprint, request, jsonify
from appService import getRatingFromModel
from constants import *

phobert = Blueprint('phobert', __name__, url_prefix="/api/v1")

@phobert.post('/predict')
def using_ensemble_cnn_bilstm():
    req = request.get_json()
    title = req.get('title')
    content = req.get('content')

    res = getRatingFromModel(title, content).tolist()

    return jsonify({'prediction': res})

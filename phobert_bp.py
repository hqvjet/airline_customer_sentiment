from flask import Blueprint, request, jsonify
from appService import getRatingFromModel
from constants import *

phobert = Blueprint('phobert', __name__, url_prefix="/api/v1/phobert")

@phobert.post('/get_predict')
def using_ensemble_cnn_bilstm():
    req = request.get_json()
    content = req.get('content')

    res = getRatingFromModel(content).tolist()

    return jsonify({'prediction': res})

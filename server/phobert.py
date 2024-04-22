from flask import Blueprint, request, jsonify
from appService import getRatingFromModel
from constants import *

phobert = Blueprint('phobert', __name__, url_prefix="/api/v1/phobert")

@phobert.post('/cnn')
def using_cnn():
    req = request.get_json()
    title = req.get('title')
    content = req.get('content')

    res = getRatingFromModel(title, content, CNN_MODEL, PHOBERT_METHOD).tolist()

    return jsonify({'prediction': res})

@phobert.post('/lstm')
def using_lstm():
    req = request.get_json()
    title = req.get('title')
    content = req.get('content')

    res = getRatingFromModel(title, content, LSTM_MODEL, PHOBERT_METHOD).tolist()

    return jsonify({'prediction': res})

@phobert.post('/bilstm')
def using_bilstm():
    req = request.get_json()
    title = req.get('title')
    content = req.get('content')

    res = getRatingFromModel(title, content, BILSTM_MODEL, PHOBERT_METHOD).tolist()

    return jsonify({'prediction': res})

@phobert.post('/ensemble_cnn_bilstm')
def using_ensemble_cnn_bilstm():
    req = request.get_json()
    title = req.get('title')
    content = req.get('content')

    res = getRatingFromModel(title, content, ENSEMBLE_CNN_BILSTM, PHOBERT_METHOD).tolist()

    return jsonify({'prediction': res})

# @phobert.post('/fusion')
# def using_fusion_cnn_bilstm():
#     req = request.get_json()
#     title = req.get('title')
#     content = req.get('content')

#     res = getRatingFromModel(title, content, FUSION_CNN_BILSTM).tolist()

#     return jsonify(res)

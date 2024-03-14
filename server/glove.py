from flask import Blueprint, request, jsonify
from appService import getRatingFromModel

glove = Blueprint('glove', __name__, url_prefix="/api/v1/glove")

@glove.post('/cnn')
def using_cnn():
    req = request.get_json()
    title = req.get('title')
    content = req.get('content')

    res = getRatingFromModel(title, content, 'CNN_MODEL.keras')

    return jsonify(res)

@glove.post('/lstm')
def using_cnn():
    req = request.get_json()
    title = req.get('title')
    content = req.get('content')

    res = getRatingFromModel(title, content, 'LSTM_MODEL.keras')

    return jsonify(res)

@glove.post('/bilstm')
def using_cnn():
    req = request.get_json()
    title = req.get('title')
    content = req.get('content')

    res = getRatingFromModel(title, content, 'BILSTM_MODEL.keras')

    return jsonify(res)

@glove.post('/ensemble')
def using_cnn():
    req = request.get_json()
    title = req.get('title')
    content = req.get('content')

    res = getRatingFromModel(title, content, 'ENSEMBLE_CNN_BILSTM_MODEL.keras')

    return jsonify(res)

@glove.post('/fusion')
def using_cnn():
    req = request.get_json()
    title = req.get('title')
    content = req.get('content')

    res = getRatingFromModel(title, content, 'FUSION_CNN_BILSTM_MODEL.keras')

    return jsonify(res)

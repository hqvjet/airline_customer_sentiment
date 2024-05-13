from flask import Blueprint, request, jsonify
from appService import getRatingFromModel
from constants import *

glove = Blueprint('glove', __name__, url_prefix="/api/v1/glove")

@glove.post('/cnn')
def using_cnn():
    req = request.get_json()
    title = req.get('title')
    content = req.get('content')

    res = getRatingFromModel(title, content, CNN_MODEL, GLOVE_METHOD).tolist()

    return jsonify({'prediction': res})

@glove.post('/lstm')
def using_lstm():
    req = request.get_json()
    title = req.get('title')
    content = req.get('content')

    res = getRatingFromModel(title, content, LSTM_MODEL, GLOVE_METHOD).tolist()

    return jsonify({'prediction': res})

@glove.post('/bilstm')
def using_bilstm():
    req = request.get_json()
    title = req.get('title')
    content = req.get('content')

    res = getRatingFromModel(title, content, BILSTM_MODEL, GLOVE_METHOD).tolist()

    return jsonify({'prediction': res})

@glove.post('/gru')
def using_gru():
    req = request.get_json()
    title = req.get('title')
    content = req.get('content')

    res = getRatingFromModel(title, content, GRU_MODEL, GLOVE_METHOD).tolist()
    return jsonify({'prediction': res})

@glove.post('/bigru')
def using_bigru():
    req = request.get_json()
    title = req.get('title')
    content = req.get('content')

    res = getRatingFromModel(title, content, BIGRU_MODEL, GLOVE_METHOD).tolist()
    return jsonify({'prediction': res})

@glove.post('/ensemble_cnn_bilstm')
def using_ensemble_cnn_bilstm():
    req = request.get_json()
    title = req.get('title')
    content = req.get('content')

    res = getRatingFromModel(title, content, ENSEMBLE_CNN_BILSTM, GLOVE_METHOD).tolist()

    return jsonify({'prediction': res})

@glove.post('/ensemble_cnn_bigru')
def using_ensemble_cnn_bigru():
    req = request.get_json()
    title = req.get('title')
    content = req.get('content')

    res = getRatingFromModel(title, content, ENSEMBLE_CNN_BIGRU, GLOVE_METHOD).tolist()

    return jsonify({'prediction': res})

@glove.post('/fusion_cnn_bilstm')
def using_fusion_cnn_bilstm():
    req = request.get_json()
    title = req.get('title')
    content = req.get('content')

    res = getRatingFromModel(title, content, FUSION_CNN_BILSTM, GLOVE_METHOD).tolist()

    return jsonify({'prediction': res})

@glove.post('/fusion_cnn_bigru')
def using_fusion_cnn_bigru():
    req = request.get_json()
    title = req.get('title')
    content = req.get('content')

    res = getRatingFromModel(title, content, FUSION_CNN_BIGRU, GLOVE_METHOD).tolist()

    return jsonify({'prediction': res})

@glove.post('/transformer')
def using_transformer():
    req = request.get_json()
    title = req.get('title')
    content = req.get('content')

    res = getRatingFromModel(title, content, TRANSFORMER_MODEL, GLOVE_METHOD).tolist()
    return jsonify({'prediction': res})

@glove.post('/random_forest')
def using_random_forest():
    req = request.get_json()
    title = req.get('title')
    content = req.get('content')

    res = getRatingFromModel(title, content, DECISION_FOREST_MODEL, GLOVE_METHOD).tolist()
    return jsonify({'prediction': res})

@glove.post('/logistic')
def using_log_reg():
    req = request.get_json()
    title = req.get('title')
    content = req.get('content')

    res = getRatingFromModel(title, content, LOGIS_REG_MODEL, GLOVE_METHOD).tolist()
    return jsonify({'prediction': res})

@glove.post('/sgd')
def using_sgd():
    req = request.get_json()
    title = req.get('title')
    content = req.get('content')

    res = getRatingFromModel(title, content, SGD_MODEL, GLOVE_METHOD).tolist()
    return jsonify({'prediction': res})


 

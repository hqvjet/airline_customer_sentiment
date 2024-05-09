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

@phobert.post('/gru')
def using_gru():
    req = request.get_json()
    title = req.get('title')
    content = req.get('content')

    res = getRatingFromModel(title, content, GRU_MODEL, PHOBERT_METHOD).tolist()
    return jsonify({'prediction': res})

@phobert.post('/bigru')
def using_bigru():
    req = request.get_json()
    title = req.get('title')
    content = req.get('content')

    res = getRatingFromModel(title, content, BIGRU_MODEL, PHOBERT_METHOD).tolist()
    return jsonify({'prediction': res})

@phobert.post('/ensemble_cnn_bilstm')
def using_ensemble_cnn_bilstm():
    req = request.get_json()
    title = req.get('title')
    content = req.get('content')

    res = getRatingFromModel(title, content, ENSEMBLE_CNN_BILSTM, PHOBERT_METHOD).tolist()

    return jsonify({'prediction': res})

@phobert.post('/ensemble_cnn_bigru')
def using_ensemble_cnn_bigru():
    req = request.get_json()
    title = req.get('title')
    content = req.get('content')

    res = getRatingFromModel(title, content, ENSEMBLE_CNN_BIGRU, PHOBERT_METHOD).tolist()

    return jsonify({'prediction': res})

@phobert.post('/fusion_cnn_bilstm')
def using_fusion_cnn_bilstm():
    req = request.get_json()
    title = req.get('title')
    content = req.get('content')

    res = getRatingFromModel(title, content, FUSION_CNN_BILSTM, PHOBERT_METHOD).tolist()

    return jsonify({'prediction': res})

@phobert.post('/fusion_cnn_bigru')
def using_fusion_cnn_bigru():
    req = request.get_json()
    title = req.get('title')
    content = req.get('content')

    res = getRatingFromModel(title, content, FUSION_CNN_BIGRU, PHOBERT_METHOD).tolist()

    return jsonify({'prediction': res})

@phobert.post('/transformer')
def using_transformer():
    req = request.get_json()
    title = req.get('title')
    content = req.get('content')

    res = getRatingFromModel(title, content, TRANSFORMER_MODEL, PHOBERT_METHOD).tolist()
    return jsonify({'prediction': res})

@phobert.post('/random_forest')
def using_rand_for():
    req = request.get_json()
    title = req.get('title')
    content = req.get('content')

    res = getRatingFromModel(title, content, DECISION_FOREST_MODEL, PHOBERT_METHOD).tolist()
    return jsonify({'prediction': res})

@phobert.post('/logistic')
def using_log_reg():
    req = request.get_json()
    title = req.get('title')
    content = req.get('content')

    res = getRatingFromModel(title, content, LOGIS_REG_MODEL, PHOBERT_METHOD).tolist()
    return jsonify({'prediction': res})

@phobert.post('/sgd')
def using_sgd():
    req = request.get_json()
    title = req.get('title')
    content = req.get('content')

    res = getRatingFromModel(title, content, SGD_MODEL, PHOBERT_METHOD).tolist()
    return jsonify(res)

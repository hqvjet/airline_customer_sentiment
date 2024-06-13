export const GLOVE_INFO = {
    CNN: {
        name: 'CNN MODEL',
        neg: 91,
        neu: 43,
        pos: 88
    },
    LSTM: {
        name: 'LSTM MODEL',
        neg: 90,
        neu: 19,
        pos: 87
    },
    BILSTM: {
        name: 'BILSTM MODEL',
        neg: 91,
        neu: 42,
        pos: 87
    },
    FUSION_CNN_BILSTM: {
        name: 'FUSION CNN MODEL',
        neg: 93,
        neu: 48,
        pos: 89
    },
    ENSEMBLE_CNN_BILSTM: {
        name: 'ENSEMBLE CNN MODEL',
        neg: 92,
        neu: 28,
        pos: 90
    },
}

export const PHOBERT_INFO = {
    CNN: {
        name: 'CNN MODEL',
        neg: 91,
        neu: 43,
        pos: 88
    },
    LSTM: {
        name: 'LSTM MODEL',
        neg: 90,
        neu: 19,
        pos: 87
    },
    BILSTM: {
        name: 'BILSTM MODEL',
        neg: 91,
        neu: 42,
        pos: 87
    },
    FUSION_CNN_BILSTM: {
        name: 'FUSION CNN MODEL',
        neg: 93,
        neu: 48,
        pos: 89
    },
    ENSEMBLE_CNN_BILSTM: {
        name: 'ENSEMBLE CNN MODEL',
        neg: 92,
        neu: 28,
        pos: 90
    },
}

// Model list
export const MODEL_LIST = {
    phobert: [
        {
            name: 'CNN MODEL',
            accuracy: '80%',
            path: 'cnn'
        },
        {
            name: 'LSTM MODEL',
            accuracy: '80%',
            path: 'lstm'
        },
        {
            name: 'BiLSTM MODEL',
            accuracy: '80%',
            path: 'bilstm'
        },
        {
            name: 'GRU MODEL',
            accuracy: '80%',
            path: 'gru'
        },
        {
            name: 'BiGRU MODEL',
            accuracy: '80%',
            path: 'bigru'
        },
        {
            name: 'ENSEMBLE CNN + BiLSTM',
            accuracy: '80%',
            path: 'ensemble_cnn_bilstm'
        },
        {
            name: 'FUSION CNN + BiLSTM',
            accuracy: '80%',
            path: 'fusion_cnn_bilstm'
        },
        {
            name: 'FUSION CNN BiGRU',
            accuracy: '80%',
            path: 'fusion_cnn_bigru'
        },
        {
            name: 'TRANSFORMER',
            accuracy: '80%',
            path: 'transformer'
        },
        {
            name: 'RANDOM FOREST',
            accuracy: '80%',
            path: 'random_forest'
        },
        {
            name: 'SGD',
            accuracy: '80%',
            path: 'sgd'
        },
        {
            name: 'LOGISTIC REGRESSION',
            accuracy: '80%',
            path: 'logistic'
        },
        {
            name: 'K-NEAREST NEIGHBORS (KNN)',
            accuracy: '80%',
            path: 'knn'
        },
        {
            name: 'SUPPORT VECTOR MACHINE (SVM)',
            accuracy: '80%',
            path: 'svm'
        },
        {
            name: 'NAIVE BAYES',
            accuracy: '80%',
            path: 'nb'
        }
    ],
    glove: [
        {
            name: 'CNN MODEL',
            accuracy: '80%',
            path: 'cnn'
        },
        {
            name: 'LSTM MODEL',
            accuracy: '80%',
            path: 'lstm'
        },
        {
            name: 'BiLSTM MODEL',
            accuracy: '80%',
            path: 'bilstm'
        },
        {
            name: 'GRU MODEL',
            accuracy: '80%',
            path: 'gru'
        },
        {
            name: 'BiGRU MODEL',
            accuracy: '80%',
            path: 'bigru'
        },
        {
            name: 'ENSEMBLE CNN + BiLSTM',
            accuracy: '80%',
            path: 'ensemble_cnn_bilstm'
        },
        {
            name: 'FUSION CNN + BiLSTM',
            accuracy: '80%',
            path: 'fusion_cnn_bilstm'
        },
        {
            name: 'FUSION CNN BiGRU',
            accuracy: '80%',
            path: 'fusion_cnn_bigru'
        },
        {
            name: 'TRANSFORMER',
            accuracy: '80%',
            path: 'transformer'
        },
        {
            name: 'RANDOM FOREST',
            accuracy: '80%',
            path: 'random_forest'
        },
        {
            name: 'SGD',
            accuracy: '80%',
            path: 'sgd'
        },
        {
            name: 'LOGISTIC REGRESSION',
            accuracy: '80%',
            path: 'logistic'
        }
    ]
}
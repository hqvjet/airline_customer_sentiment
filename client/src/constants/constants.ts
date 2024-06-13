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
            accuracy: '89%',
            path: 'cnn'
        },
        {
            name: 'LSTM MODEL',
            accuracy: '88%',
            path: 'lstm'
        },
        {
            name: 'BiLSTM MODEL',
            accuracy: '88%',
            path: 'bilstm'
        },
        {
            name: 'GRU MODEL',
            accuracy: '88%',
            path: 'gru'
        },
        {
            name: 'BiGRU MODEL',
            accuracy: '88%',
            path: 'bigru'
        },
        {
            name: 'ENSEMBLE CNN + BiLSTM',
            accuracy: '96%',
            path: 'ensemble_cnn_bilstm'
        },
        {
            name: 'FUSION CNN + BiLSTM',
            accuracy: '89%',
            path: 'fusion_cnn_bilstm'
        },
        {
            name: 'FUSION CNN BiGRU',
            accuracy: '87%',
            path: 'fusion_cnn_bigru'
        },
        {
            name: 'TRANSFORMER',
            accuracy: '92%',
            path: 'transformer'
        },
        {
            name: 'RANDOM FOREST',
            accuracy: '76%',
            path: 'random_forest'
        },
        {
            name: 'SGD',
            accuracy: '88%',
            path: 'sgd'
        },
        {
            name: 'LOGISTIC REGRESSION',
            accuracy: '97%',
            path: 'logistic'
        },
        {
            name: 'K-NEAREST NEIGHBORS (KNN)',
            accuracy: '83%',
            path: 'knn'
        },
        {
            name: 'SUPPORT VECTOR MACHINE (SVM)',
            accuracy: '82%',
            path: 'svm'
        },
        {
            name: 'NAIVE BAYES',
            accuracy: '68%',
            path: 'nb'
        }
    ],
    glove: [
        {
            name: 'CNN MODEL',
            accuracy: '90%',
            path: 'cnn'
        },
        {
            name: 'LSTM MODEL',
            accuracy: '88%',
            path: 'lstm'
        },
        {
            name: 'BiLSTM MODEL',
            accuracy: '89%',
            path: 'bilstm'
        },
        {
            name: 'GRU MODEL',
            accuracy: '85%',
            path: 'gru'
        },
        {
            name: 'BiGRU MODEL',
            accuracy: '86%',
            path: 'bigru'
        },
        {
            name: 'ENSEMBLE CNN + BiLSTM',
            accuracy: '93%',
            path: 'ensemble_cnn_bilstm'
        },
        {
            name: 'FUSION CNN + BiLSTM',
            accuracy: '86%',
            path: 'fusion_cnn_bilstm'
        },
        {
            name: 'FUSION CNN BiGRU',
            accuracy: '87%',
            path: 'fusion_cnn_bigru'
        },
        {
            name: 'TRANSFORMER',
            accuracy: '86%',
            path: 'transformer'
        },
        {
            name: 'RANDOM FOREST',
            accuracy: '73%',
            path: 'random_forest'
        },
        {
            name: 'SGD',
            accuracy: '84%',
            path: 'sgd'
        },
        {
            name: 'LOGISTIC REGRESSION',
            accuracy: '79%',
            path: 'logistic'
        }
    ]
}
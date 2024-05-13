import { GLOVE_CONFIG } from ".";

export const usingGLOVE = {
    usingCNN: (title: string, content: string) => {
        return GLOVE_CONFIG.post('/cnn', {title: title, content: content})
    },
    usingLSTM: (title: string, content: string) => {
        return GLOVE_CONFIG.post('/lstm', {title: title, content: content})
    },
    usingBILSTM: (title: string, content: string) => {
        return GLOVE_CONFIG.post('/bilstm', {title: title, content: content})
    },
    usingENSEMBLE_CNN_BILSTM: (title: string, content: string) => {
        return GLOVE_CONFIG.post('/ensemble_cnn_bilstm', {title: title, content: content})
    },
    usingFUSION_CNN_BILSTM: (title: string, content: string) => {
        return GLOVE_CONFIG.post('/fusion_cnn_bilstm', {title: title, content: content})
    },
    usingENSEMBLE_CNN_BIGRU: (title: string, content: string) => {
        return GLOVE_CONFIG.post('/ensemble_cnn_bigru', {title: title, content: content})
    },
    usingFUSION_CNN_BIGRU: (title: string, content: string) => {
        return GLOVE_CONFIG.post('/fusion_cnn_bigru', {title: title, content: content})
    },
    usingGRU: (title: string, content: string) => {
        return GLOVE_CONFIG.post('/gru', {title: title, content: content})
    },
    usingBIGRU: (title: string, content: string) => {
        return GLOVE_CONFIG.post('/bigru', {title: title, content: content})
    },
    usingTRANSFORMER: (title: string, content: string) => {
        return GLOVE_CONFIG.post('/transformer', {title: title, content: content})
    },
    usingLOGISTIC: (title: string, content: string) => {
        return GLOVE_CONFIG.post('/logistic', {title: title, content: content})
    },
    usingSGD: (title: string, content: string) => {
        return GLOVE_CONFIG.post('/sgd', {title: title, content: content})
    },
    usingRANDOM_FOREST: (title: string, content: string) => {
        return GLOVE_CONFIG.post('/random_forest', {title: title, content: content})
    }
}

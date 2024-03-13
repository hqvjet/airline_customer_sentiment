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
}
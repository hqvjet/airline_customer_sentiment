import axios from 'axios'
import { usingGLOVE } from './glove'
import { usingPHOBERT } from './phobert'

export const GLOVE_CONFIG = axios.create({
    baseURL: process.env.NEXT_PUBLIC_API_URL + '/glove',
    timeout: 10000
})

export const PHOBERT_CONFIG = axios.create({
    baseURL: process.env.NEXT_PUBLIC_API_URL + '/phobert',
    timeout: 10000
})

export const JSVCONFIG = axios.create({
    baseURL: 'http://192.168.17.191:8000',
    timeout: 10000
})

export default function useAPI(path: string) {
    const paths = path.split('/')
    const emb = paths[2]
    const model = paths[3]

    if (emb == 'glove') {
        if (model == 'cnn')
            return usingGLOVE.usingCNN
        else if (model == 'lstm')
            return usingGLOVE.usingLSTM
        else if (model == 'bilstm')
            return usingGLOVE.usingBILSTM
        else if (model == 'gru')
            return usingGLOVE.usingGRU
        else if (model == 'bigru')
            return usingGLOVE.usingBIGRU
        else if (model == 'ensemble_cnn_bilstm')
            return usingGLOVE.usingENSEMBLE_CNN_BILSTM
        else if (model == 'ensemble_cnn_bigru')
            return usingGLOVE.usingENSEMBLE_CNN_BIGRU
        else if (model == 'fusion_cnn_bilstm')
            return usingGLOVE.usingFUSION_CNN_BILSTM
        else if (model == 'fusion_cnn_bigru')
            return usingGLOVE.usingFUSION_CNN_BIGRU
        else if (model == 'transformer')
            return usingGLOVE.usingTRANSFORMER
        else if (model == 'logistic')
            return usingGLOVE.usingLOGISTIC
        else if (model == 'sgd')
            return usingGLOVE.usingSGD
        else if (model == 'random_forest')
            return usingGLOVE.usingRANDOM_FOREST
    }
    else if (emb == 'phobert') {
        if (model == 'cnn') 
            return usingPHOBERT.usingCNN
        else if (model == 'lstm')
            return usingPHOBERT.usingLSTM
        else if (model == 'bilstm')
            return usingPHOBERT.usingBILSTM
        else if (model == 'gru')
            return usingPHOBERT.usingGRU
        else if (model == 'bigru')
            return usingPHOBERT.usingBIGRU
        else if (model == 'ensemble_cnn_bilstm')
            return usingPHOBERT.usingENSEMBLE_CNN_BILSTM
        else if (model == 'ensemble_cnn_bigru')
            return usingPHOBERT.usingENSEMBLE_CNN_BIGRU
        else if (model == 'fusion_cnn_bilstm')
            return usingPHOBERT.usingFUSION_CNN_BILSTM
        else if (model == 'fusion_cnn_bigru')
            return usingPHOBERT.usingFUSION_CNN_BIGRU
        else if (model == 'transformer')
            return usingPHOBERT.usingTRANSFORMER
        else if (model == 'logistic')
            return usingPHOBERT.usingLOGISTIC
        else if (model == 'sgd')
            return usingPHOBERT.usingSGD
        else if (model == 'random_forest')
            return usingPHOBERT.usingRANDOM_FOREST
        else if (model == 'knn')
            return usingPHOBERT.usingKNN
        else if (model == 'svm')
            return usingPHOBERT.usingSVM
        else if (model == 'nb')
            return usingPHOBERT.usingNB
    }

    return null
}

import axios from 'axios'
import { usingGLOVE } from './glove'
import { usingPHOBERT } from './phobert'

export const GLOVE_CONFIG = axios.create({
    baseURL: process.env.SERVER_API + '/glove',
    timeout: 10000
})

export const PHOBERT_CONFIG = axios.create({
    baseURL: process.env.SERVER_API + '/phobert',
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
        else if (model == 'ensemble_cnn_bilstm')
            return usingGLOVE.usingENSEMBLE_CNN_BILSTM
        else if ( model == 'fusion_cnn_bilstm')
            return usingGLOVE.usingFUSION_CNN_BILSTM
    }
    else if (emb == 'phobert') {
        if (model == 'cnn') 
            return usingPHOBERT.usingCNN
        else if (model == 'lstm')
            return usingPHOBERT.usingLSTM
        else if (model == 'bilstm')
            return usingPHOBERT.usingBILSTM
        else if (model == 'ensemble_cnn_bilstm')
            return usingPHOBERT.usingENSEMBLE_CNN_BILSTM
        else if ( model == 'fusion_cnn_bilstm')
            return usingPHOBERT.usingFUSION_CNN_BILSTM
    }

    return null
}
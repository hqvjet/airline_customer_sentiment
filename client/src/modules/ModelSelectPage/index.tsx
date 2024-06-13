'use client'
import { Col, Row, Space, Card, Button, Typography } from "antd"
import Link from 'next/link';
import { GLOVE_INFO, PHOBERT_INFO, MODEL_LIST } from '@/constants/constants'

const { Title, Text } = Typography
export default function ModelSelectPage() {

    return (
        <Col className="w-full h-auto text-center scroll-my-6">
            <Title level={1}><p className="text-gray-800 font-extrabold">AIRLINE CUSTOMER SENTIMENT</p></Title>
            <Col className="w-full h-1/2 text-center flex flex-col">
                <Title><p className="text-black-800 text-3xl text-gray-50 font-extrabold">PHOBERT EMBEDDING</p></Title>
                <Row className="w-full h-auto gap-3 -mt-5 mb-10" justify={'center'}>
                    {MODEL_LIST.phobert.map(model => (
                        <Link href={"/show-model/phobert/" + model.path} className="w-1/5" passHref>
                            <Button size='large' type='primary' className="w-full h-full shadow shadow-white bg-gray-800 bg-opacity-80" >
                                <p className="text-white">{model.name}</p>
                                <p className="text-white">Accuracy: {model.accuracy}</p>
                            </Button>
                        </Link>
                    ))}
                </Row>
            </Col>
            <Col className="w-full h-1/2 text-center flex flex-col">
                <Title><p className="text-black-800 text-3xl text-gray-50 font-extrabold">GLOVE EMBEDDING</p></Title>
                <Row className="w-full h-auto gap-3 -mt-5 mb-10" justify={'center'}>
                    {MODEL_LIST.glove.map(model => (
                        <Link href={"/show-model/phobert/" + model.path} className="w-1/5" passHref>
                            <Button size='large' type='primary' className="w-full h-full shadow shadow-white bg-gray-800 bg-opacity-80" >
                                <p className="text-white">{model.name}</p>
                                <p className="text-white">Accuracy: {model.accuracy}</p>
                            </Button>
                        </Link>
                    ))}
                </Row>
            </Col>
        </Col>

        // <Col className="w-full h-full flex flex-col justify-center items-center align-middle">
        //     <Title level={1}><p className="text-white">AIRLINE COMMENT SENTIMENT DEEP LEARNING MODELS</p></Title>
        //     <Col className="w-full h-1/2 flex flex-col justify-center items-center">
        //         <Title><p className="text-white text-center">Global Vector Embedding</p></Title>
        //         <Row className="w-full h-32 gap-5" justify={'center'}>
        //             <Link href="/show-model/glove/cnn" className="w-1/6" passHref>
        //                 <Button size='large' type='primary' className="w-full h-full shadow shadow-white" >
        //                     <p className="text-white">CNN MODEL</p>
        //                 </Button>
        //             </Link>
        //             <Link href="/show-model/glove/lstm" className="w-1/6" passHref>
        //                 <Button size='large' type='primary' className="w-full h-full shadow shadow-white" >
        //                     <p className="text-white">{GLOVE_INFO.LSTM.name}</p>
        //                 </Button>
        //             </Link>
        //             <Link href="/show-model/glove/bilstm" className="w-1/6" passHref>
        //                 <Button size='large' type='primary' className="w-full h-full shadow shadow-white" >
        //                     <p className="text-white">BiLSTM MODEL</p>
        //                 </Button>
        //             </Link>
        //             <Link href="/show-model/glove/gru" className="w-1/6" passHref>
        //                 <Button size='large' type='primary' className="w-full h-full shadow shadow-white" >
        //                     <p className="text-white">GRU</p>
        //                 </Button>
        //             </Link>
        //             <Link href="/show-model/glove/bigru" className="w-1/6" passHref>
        //                 <Button size='large' type='primary' className="w-full h-full shadow shadow-white" >
        //                     <p className="text-white">BiGRU</p>
        //                 </Button>
        //             </Link>
        //             <Link href="/show-model/glove/ensemble_cnn_bilstm" className="w-1/6" passHref>
        //                 <Button size='large' type='primary' className="w-full h-full shadow shadow-white" >
        //                     <p className="text-white">ENSEMBLE CNN + BILSTM</p>
        //                 </Button>
        //             </Link>
        //             <Link href="/show-model/glove/fusion_cnn_bilstm" className="w-1/6" passHref>
        //                 <Button size='large' type='primary' className="w-full h-full shadow shadow-white" >
        //                     <p className="text-white">FUSION CNN + BILSTM</p>
        //                 </Button>
        //             </Link>
        //             <Link href="/show-model/glove/fusion_cnn_bigru" className="w-1/6" passHref>
        //                 <Button size='large' type='primary' className="w-full h-full shadow shadow-white" >
        //                     <p className="text-white">FUSION CNN BiGRU</p>
        //                 </Button>
        //             </Link>
        //             <Link href="/show-model/glove/transformer" className="w-1/6" passHref>
        //                 <Button size='large' type='primary' className="w-full h-full shadow shadow-white" >
        //                     <p className="text-white">TRANSFORMER</p>
        //                 </Button>
        //             </Link>
        //             <Link href="/show-model/glove/random_forest" className="w-1/6" passHref>
        //                 <Button size='large' type='primary' className="w-full h-full shadow shadow-white" >
        //                     <p className="text-white">RANDOM FOREST</p>
        //                 </Button>
        //             </Link>
        //             <Link href="/show-model/glove/sgd" className="w-1/6" passHref>
        //                 <Button size='large' type='primary' className="w-full h-full shadow shadow-white" >
        //                     <p className="text-white">SGD</p>
        //                 </Button>
        //             </Link>
        //             <Link href="/show-model/glove/random_forest" className="w-1/6" passHref>
        //                 <Button size='large' type='primary' className="w-full h-full shadow shadow-white" >
        //                     <p className="text-white">KNN</p>
        //                 </Button>
        //             </Link>
        //        </Row >
        //     </Col >
        //     <Col className="w-full h-1/2 flex flex-col justify-center items-center">
        //         <Title><p className="text-white text-center">PhoBERT Embedding</p></Title>
        //         <Row className="w-full h-32 gap-5" justify={'center'}>
        //             <Link href="/show-model/phobert/cnn" className="w-1/6" passHref>
        //                 <Button size='large' type='primary' className="w-full h-full shadow shadow-white" >
        //                     <p className="text-white">CNN MODEL</p>
        //                 </Button>
        //             </Link>
        //             <Link href="/show-model/phobert/lstm" className="w-1/6" passHref>
        //                 <Button size='large' type='primary' className="w-full h-full shadow shadow-white" >
        //                     <p className="text-white">LSTM MODEL</p>
        //                 </Button>
        //             </Link>
        //             <Link href="/show-model/phobert/bilstm" className="w-1/6" passHref>
        //                 <Button size='large' type='primary' className="w-full h-full shadow shadow-white" >
        //                     <p className="text-white">BiLSTM MODEL</p>
        //                 </Button>
        //             </Link>
        //             <Link href="/show-model/phobert/gru" className="w-1/6" passHref>
        //                 <Button size='large' type='primary' className="w-full h-full shadow shadow-white" >
        //                     <p className="text-white">GRU</p>
        //                 </Button>
        //             </Link>
        //             <Link href="/show-model/phobert/bigru" className="w-1/6" passHref>
        //                 <Button size='large' type='primary' className="w-full h-full shadow shadow-white" >
        //                     <p className="text-white">BIGRU</p>
        //                 </Button>
        //             </Link>
        //             <Link href="/show-model/phobert/ensemble_cnn_bilstm" className="w-1/6" passHref>
        //                 <Button size='large' type='primary' className="w-full h-full shadow shadow-white" >
        //                     <p className="text-white">ENSEMBLE CNN + BILSTM</p>
        //                 </Button>
        //             </Link>
        //             <Link href="/show-model/phobert/fusion_cnn_bilstm" className="w-1/6" passHref>
        //                 <Button size='large' type='primary' className="w-full h-full shadow shadow-white" >
        //                     <p className="text-white">FUSION CNN + BILSTM</p>
        //                 </Button>
        //             </Link>
        //             <Link href="/show-model/phobert/fusion_cnn_bigru" className="w-1/6" passHref>
        //                 <Button size='large' type='primary' className="w-full h-full shadow shadow-white" >
        //                     <p className="text-white">FUSION CNN + BIGRU</p>
        //                 </Button>
        //             </Link>
        //             <Link href="/show-model/phobert/transformer" className="w-1/6" passHref>
        //                 <Button size='large' type='primary' className="w-full h-full shadow shadow-white" >
        //                     <p className="text-white">TRANSFORMER</p>
        //                 </Button>
        //             </Link>
        //             <Link href="/show-model/phobert/random_forest" className="w-1/6" passHref>
        //                 <Button size='large' type='primary' className="w-full h-full shadow shadow-white" >
        //                     <p className="text-white">RANDOM FOREST</p>
        //                 </Button>
        //             </Link>
        //             <Link href="/show-model/phobert/sgd" className="w-1/6" passHref>
        //                 <Button size='large' type='primary' className="w-full h-full shadow shadow-white" >
        //                     <p className="text-white">SGD</p>
        //                 </Button>
        //             </Link>
        //             <Link href="/show-model/phobert/random_forest" className="w-1/6" passHref>
        //                 <Button size='large' type='primary' className="w-full h-full shadow shadow-white" >
        //                     <p className="text-white">KNN</p>
        //                 </Button>
        //             </Link>
        //         </Row>
        //     </Col>
        // </Col >
    )
}

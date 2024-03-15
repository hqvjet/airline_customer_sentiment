'use client'
import { Col, Row, Space, Card, Button, Typography } from "antd"
import Link from 'next/link';
import { GLOVE_INFO, PHOBERT_INFO } from '@/constants/constants'

const { Title, Text } = Typography
export default function ModelSelectPage() {

    return (
        <Col className="w-full h-full flex flex-col justify-center items-center align-middle">
            <Title level={1}><p className="text-white">AIRLINE COMMENT SENTIMENT DEEP LEARNING MODELS</p></Title>
            <Col className="w-full h-1/2 flex flex-col justify-center items-center">
                <Title><p className="text-white text-center">Global Vector Embedding</p></Title>
                <Row className="w-full h-32 gap-5" justify={'center'}>
                    <Link href="/show-model/glove/lstm" className="w-1/6" passHref>
                        <Button size='large' type='primary' className="w-full h-full shadow shadow-white" >
                            <p className="text-white">{GLOVE_INFO.LSTM.name}</p>
                            <Col>
                                <p className="text-green-500">Positive: {GLOVE_INFO.LSTM.pos}</p>
                                <p className="text-yellow-500">Neutral: {GLOVE_INFO.LSTM.neu}</p>
                                <p className="text-red-500">Negative: {GLOVE_INFO.LSTM.neg}</p>
                            </Col>
                        </Button>
                    </Link>
                    <Link href="/show-model/glove/cnn" className="w-1/6" passHref>
                        <Button size='large' type='primary' className="w-full h-full shadow shadow-white" >
                            <p className="text-white">CNN MODEL</p>
                            <Col>
                                <p className="text-green-500">Positive: {GLOVE_INFO.CNN.pos}</p>
                                <p className="text-yellow-500">Neutral: {GLOVE_INFO.CNN.neu}</p>
                                <p className="text-red-500">Negative: {GLOVE_INFO.CNN.neg}</p>
                            </Col>
                        </Button>
                    </Link>
                    <Link href="/show-model/glove/bilstm" className="w-1/6" passHref>
                        <Button size='large' type='primary' className="w-full h-full shadow shadow-white" >
                            <p className="text-white">BiLSTM MODEL</p>
                            <Col>
                                <p className="text-green-500">Positive: {GLOVE_INFO.BILSTM.pos}</p>
                                <p className="text-yellow-500">Neutral: {GLOVE_INFO.BILSTM.neu}</p>
                                <p className="text-red-500">Negative: {GLOVE_INFO.BILSTM.neg}</p>
                            </Col>
                        </Button>
                    </Link>
                    <Link href="/show-model/glove/ensemble_cnn_bilstm" className="w-1/6" passHref>
                        <Button size='large' type='primary' className="w-full h-full shadow shadow-white" >
                            <p className="text-white">ENSEMBLE CNN + BILSTM</p>
                            <Col>
                                <p className="text-green-500">Positive: {GLOVE_INFO.ENSEMBLE_CNN_BILSTM.pos}</p>
                                <p className="text-yellow-500">Neutral: {GLOVE_INFO.ENSEMBLE_CNN_BILSTM.neu}</p>
                                <p className="text-red-500">Negative: {GLOVE_INFO.ENSEMBLE_CNN_BILSTM.neg}</p>
                            </Col>
                        </Button>
                    </Link>
                    <Link href="/show-model/glove/fusion_cnn_bilstm" className="w-1/6" passHref>
                        <Button size='large' type='primary' className="w-full h-full shadow shadow-white" >
                            <p className="text-white">FUSION CNN + BILSTM</p>
                            <Col>
                                <p className="text-green-500">Positive: {GLOVE_INFO.FUSION_CNN_BILSTM.pos}</p>
                                <p className="text-yellow-500">Neutral: {GLOVE_INFO.FUSION_CNN_BILSTM.neu}</p>
                                <p className="text-red-500">Negative: {GLOVE_INFO.FUSION_CNN_BILSTM.neg}</p>
                            </Col>
                        </Button>
                    </Link>
                </Row >
            </Col >
            <Col className="w-full h-1/2 flex flex-col justify-center items-center">
                <Title><p className="text-white text-center">PhoBERT Embedding</p></Title>
                <Row className="w-full h-32 gap-5" justify={'center'}>
                    <Link href="/show-model/phobert/lstm" className="w-1/6" passHref>
                        <Button size='large' type='primary' className="w-full h-full shadow shadow-white" >
                            <p className="text-white">LSTM MODEL</p>
                            <Col>
                                <p className="text-green-500">Positive: {GLOVE_INFO.LSTM.pos}</p>
                                <p className="text-yellow-500">Neutral: {GLOVE_INFO.LSTM.neu}</p>
                                <p className="text-red-500">Negative: {GLOVE_INFO.LSTM.neg}</p>
                            </Col>
                        </Button>
                    </Link>
                    <Link href="/show-model/phobert/cnn" className="w-1/6" passHref>
                        <Button size='large' type='primary' className="w-full h-full shadow shadow-white" >
                            <p className="text-white">CNN MODEL</p>
                            <Col>
                                <p className="text-green-500">Positive: {GLOVE_INFO.CNN.pos}</p>
                                <p className="text-yellow-500">Neutral: {GLOVE_INFO.CNN.neu}</p>
                                <p className="text-red-500">Negative: {GLOVE_INFO.CNN.neg}</p>
                            </Col>
                        </Button>
                    </Link>
                    <Link href="/show-model/phobert/bilstm" className="w-1/6" passHref>
                        <Button size='large' type='primary' className="w-full h-full shadow shadow-white" >
                            <p className="text-white">BiLSTM MODEL</p>
                            <Col>
                                <p className="text-green-500">Positive: {GLOVE_INFO.BILSTM.pos}</p>
                                <p className="text-yellow-500">Neutral: {GLOVE_INFO.BILSTM.neu}</p>
                                <p className="text-red-500">Negative: {GLOVE_INFO.BILSTM.neg}</p>
                            </Col>
                        </Button>
                    </Link>
                    <Link href="/show-model/phobert/ensemble_cnn_bilstm" className="w-1/6" passHref>
                        <Button size='large' type='primary' className="w-full h-full shadow shadow-white" >
                            <p className="text-white">ENSEMBLE CNN + BILSTM</p>
                            <Col>
                                <p className="text-green-500">Positive: {GLOVE_INFO.ENSEMBLE_CNN_BILSTM.pos}</p>
                                <p className="text-yellow-500">Neutral: {GLOVE_INFO.ENSEMBLE_CNN_BILSTM.neu}</p>
                                <p className="text-red-500">Negative: {GLOVE_INFO.ENSEMBLE_CNN_BILSTM.neg}</p>
                            </Col>
                        </Button>
                    </Link>
                    <Link href="/show-model/phobert/fusion_cnn_bilstm" className="w-1/6" passHref>
                        <Button size='large' type='primary' className="w-full h-full shadow shadow-white" >
                            <p className="text-white">FUSION CNN + BILSTM</p>
                            <Col>
                                <p className="text-green-500">Positive: {GLOVE_INFO.FUSION_CNN_BILSTM.pos}</p>
                                <p className="text-yellow-500">Neutral: {GLOVE_INFO.FUSION_CNN_BILSTM.neu}</p>
                                <p className="text-red-500">Negative: {GLOVE_INFO.FUSION_CNN_BILSTM.neg}</p>
                            </Col>
                        </Button>
                    </Link>
                </Row>
            </Col>
        </Col >
    )
}
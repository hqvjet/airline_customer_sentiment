'use client'
import { Col, Row, Space, Card, Button, Typography, Layout } from "antd"
import Link from 'next/link';
import { GLOVE_INFO, PHOBERT_INFO } from '@/constants/constants'

const { Title, Text } = Typography
const { Content } = Layout
export default function ModelSelectPage() {
    const backgroundImageStyle = {
    backgroundImage: 'url("/bg.png")',
    backgroundSize: 'cover',
    backgroundPosition: 'center',
    backgroundRepeat: 'no-repeat',
    height: '100vh',
    }

    const buttonStyle = {
    backgroundColor: 'rgba(255, 255, 255, 0.8)',
    color: 'black',
    borderRadius: '4px',
    fontSize: '26px',
    fontWeight: 'bold',
    display: 'flex',
    alignItems: 'center',
    }

    return (
            <Content style={backgroundImageStyle}
                className='justify-center p-10'
            >
                <Title level={1}><p className="text-black text-center">AIRLINE COMMENT SENTIMENT DEEP LEARNING MODELS</p></Title>
                <Col className="w-full h-1/2 flex flex-col justify-center items-center">
                    <Title><p className="text-white text-center">Global Vector Embedding</p></Title>
                    <Row className="w-full h-32 gap-5" justify={'center'}>
                        <Link href="/show-model/glove/cnn" className="w-1/4" passHref>
                            <Button style={buttonStyle} className="flex flex-col h-auto px-10">
                                <p>CNN</p>
                                <p className="ml-2">Accuracy: 90%</p>
                            </Button>
                        </Link>
                        <Link href="/show-model/glove/lstm" className="w-1/4" passHref>
                            <Button style={buttonStyle} className="flex flex-col h-auto px-10">
                                <p>LSTM</p>
                                <p className="ml-2">Accuracy: 90%</p>
                            </Button>
                        </Link>
                        <Link href="/show-model/glove/bilstm" className="w-1/4" passHref>
                            <Button style={buttonStyle} className="flex flex-col h-auto px-10">
                                <p>BiLSTM</p>
                                <p className="ml-2">Accuracy: 90%</p>
                            </Button>
                        </Link>
                        <Link href="/show-model/glove/gru" className="w-1/4" passHref>
                            <Button style={buttonStyle} className="flex flex-col h-auto px-10">
                                <p>GRU</p>
                                <p className="ml-2">Accuracy: 90%</p>
                            </Button>
                        </Link>
                        <Link href="/show-model/glove/bigru" className="w-1/4" passHref>
                            <Button style={buttonStyle} className="flex flex-col h-auto px-10">
                                <p>BiGRU</p>
                                <p className="ml-2">Accuracy: 90%</p>
                            </Button>
                        </Link>
                        <Link href="/show-model/glove/ensemble_cnn_bilstm" className="w-1/4" passHref>
                            <Button style={buttonStyle} className="flex flex-col h-auto px-10">
                                <p className="h-auto">Ensemble CNN + BiLSTM</p>
                                <p className="ml-2">Accuracy: 90%</p>
                            </Button>
                        </Link>
                        <Link href="/show-model/glove/fusion_cnn_bilstm" className="w-1/4" passHref>
                            <Button style={buttonStyle} className="flex flex-col h-auto px-10">
                                <p>Fusion CNN + BiLSTM</p>
                                <p className="ml-2">Accuracy: 90%</p>
                            </Button>
                        </Link>
                        <Link href="/show-model/glove/ensemble_cnn_bigru" className="w-1/4" passHref>
                            <Button style={buttonStyle} className="flex flex-col h-auto px-10">
                                <p>Ensemble CNN + BiGRU</p>
                                <p className="ml-2">Accuracy: 90%</p>
                            </Button>
                        </Link>
                        <Link href="/show-model/glove/fusion_cnn_bigru" className="w-1/4" passHref>
                            <Button style={buttonStyle} className="flex flex-col h-auto px-10">
                                <p>Fusion CNN + BiGRU</p>
                                <p className="ml-2">Accuracy: 90%</p>
                            </Button>
                        </Link>
                        <Link href="/show-model/glove/transformer" className="w-1/4" passHref>
                            <Button style={buttonStyle} className="flex flex-col h-auto px-10">
                                <p>Transformer</p>
                                <p className="ml-2">Accuracy: 90%</p>
                            </Button>
                        </Link>
                        <Link href="/show-model/glove/random_forest" className="w-1/4" passHref>
                            <Button style={buttonStyle} className="flex flex-col h-auto px-10">
                                <p>Random Forest</p>
                                <p className="ml-2">Accuracy: 90%</p>
                            </Button>
                        </Link>
                        <Link href="/show-model/glove/sgd" className="w-1/4" passHref>
                            <Button style={buttonStyle} className="flex flex-col h-auto px-10">
                                <p>SGD</p>
                                <p className="ml-2">Accuracy: 90%</p>
                            </Button>
                        </Link>
                        <Link href="/show-model/glove/random_forest" className="w-1/4" passHref>
                            <Button style={buttonStyle} className="flex flex-col h-auto px-10">
                                <p>KNN</p>
                                <p className="ml-2">Accuracy: 90%</p>
                            </Button>
                        </Link>
                   </Row >
                </Col >
                <Col className="w-full h-1/2 flex flex-col justify-center items-center">
                    <Title><p className="text-white text-center">PhoBERT Embedding</p></Title>
                    <Row className="w-full h-32 gap-5" justify={'center'}>
                        <Link href="/show-model/phobert/cnn" className="w-1/4" passHref>
                            <Button size='large' type='primary' className="w-full h-full shadow shadow-white" >
                                <p className="text-white">CNN MODEL</p>
                            </Button>
                        </Link>
                        <Link href="/show-model/phobert/lstm" className="w-1/4" passHref>
                            <Button size='large' type='primary' className="w-full h-full shadow shadow-white" >
                                <p className="text-white">LSTM MODEL</p>
                            </Button>
                        </Link>
                        <Link href="/show-model/phobert/bilstm" className="w-1/4" passHref>
                            <Button size='large' type='primary' className="w-full h-full shadow shadow-white" >
                                <p className="text-white">BiLSTM MODEL</p>
                            </Button>
                        </Link>
                        <Link href="/show-model/phobert/gru" className="w-1/4" passHref>
                            <Button size='large' type='primary' className="w-full h-full shadow shadow-white" >
                                <p className="text-white">GRU</p>
                            </Button>
                        </Link>
                        <Link href="/show-model/phobert/bigru" className="w-1/4" passHref>
                            <Button size='large' type='primary' className="w-full h-full shadow shadow-white" >
                                <p className="text-white">BIGRU</p>
                            </Button>
                        </Link>
                        <Link href="/show-model/phobert/ensemble_cnn_bilstm" className="w-1/4" passHref>
                            <Button size='large' type='primary' className="w-full h-full shadow shadow-white" >
                                <p className="text-white">ENSEMBLE CNN + BILSTM</p>
                            </Button>
                        </Link>
                        <Link href="/show-model/phobert/fusion_cnn_bilstm" className="w-1/4" passHref>
                            <Button size='large' type='primary' className="w-full h-full shadow shadow-white" >
                                <p className="text-white">FUSION CNN + BILSTM</p>
                            </Button>
                        </Link>
                        <Link href="/show-model/phobert/fusion_cnn_bigru" className="w-1/4" passHref>
                            <Button size='large' type='primary' className="w-full h-full shadow shadow-white" >
                                <p className="text-white">FUSION CNN + BIGRU</p>
                            </Button>
                        </Link>
                        <Link href="/show-model/phobert/transformer" className="w-1/4" passHref>
                            <Button size='large' type='primary' className="w-full h-full shadow shadow-white" >
                                <p className="text-white">TRANSFORMER</p>
                            </Button>
                        </Link>
                        <Link href="/show-model/phobert/random_forest" className="w-1/4" passHref>
                            <Button size='large' type='primary' className="w-full h-full shadow shadow-white" >
                                <p className="text-white">RANDOM FOREST</p>
                            </Button>
                        </Link>
                        <Link href="/show-model/phobert/sgd" className="w-1/4" passHref>
                            <Button size='large' type='primary' className="w-full h-full shadow shadow-white" >
                                <p className="text-white">SGD</p>
                            </Button>
                        </Link>
                        <Link href="/show-model/phobert/random_forest" className="w-1/4" passHref>
                            <Button size='large' type='primary' className="w-full h-full shadow shadow-white" >
                                <p className="text-white">KNN</p>
                            </Button>
                        </Link>
                    </Row>
                </Col>
            </Content>
    )
}

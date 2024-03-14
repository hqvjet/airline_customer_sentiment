'use client'
import { Col, Row, Space, Card, Button, Typography } from "antd"
import Link from 'next/link';

const { Title, Text } = Typography
export default function ModelSelectPage() {

    return (
        <Col className="w-full h-full flex flex-col justify-center items-center align-middle">
            <Title level={1}><p className="text-white">AIRLINE COMMENT SENTIMENT DEEP LEARNING MODELS</p></Title>
            <Col className="w-full h-1/2 flex flex-col justify-center items-center">
                <Title><p className="text-white text-center">Global Vector Embedding</p></Title>
                <Row className="w-full h-32 gap-5" justify={'center'}>
                    <Link href="/show-model/glove/lstm" className="w-1/6" passHref>
                        <Button size='large' type='primary' className="w-full h-full shadow shadow-white" ><p className="text-white">LSTM MODEL</p></Button>
                    </Link>
                    <Link href="/show-model/glove/cnn" className="w-1/6" passHref>
                        <Button size='large' type='primary' className="w-full h-full shadow shadow-white" ><p className="text-white">CNN MODEL</p></Button>
                    </Link>
                    <Link href="/show-model/glove/bilstm" className="w-1/6" passHref>
                        <Button size='large' type='primary' className="w-full h-full shadow shadow-white" ><p className="text-white">BiLSTM MODEL</p></Button>
                    </Link>
                    <Link href="/show-model/glove/ensemble_cnn_bilstm" className="w-1/6" passHref>
                        <Button size='large' type='primary' className="w-full h-full shadow shadow-white" ><p className="text-white">ENSEMBLE CNN + BILSTM</p></Button>
                    </Link>
                    <Link href="/show-model/glove/fusion_cnn_bilstm" className="w-1/6" passHref>
                        <Button size='large' type='primary' className="w-full h-full shadow shadow-white" ><p className="text-white">FUSION CNN + BILSTM</p></Button>
                    </Link>
                </Row >
            </Col >
            <Col className="w-full h-1/2 flex flex-col justify-center items-center">
                <Title><p className="text-white text-center">PhoBERT Embedding</p></Title>
                <Row className="w-full h-32 gap-5" justify={'center'}>
                    <Link href="/show-model/phobert/lstm" className="w-1/6" passHref>
                        <Button size='large' type='primary' className="w-full h-full shadow shadow-white" ><p className="text-white">LSTM MODEL</p></Button>
                    </Link>
                    <Link href="/show-model/phobert/cnn" className="w-1/6" passHref>
                        <Button size='large' type='primary' className="w-full h-full shadow shadow-white" ><p className="text-white">CNN MODEL</p></Button>
                    </Link>
                    <Link href="/show-model/phobert/bilstm" className="w-1/6" passHref>
                        <Button size='large' type='primary' className="w-full h-full shadow shadow-white" ><p className="text-white">BiLSTM MODEL</p></Button>
                    </Link>
                    <Link href="/show-model/phobert/ensemble_cnn_bilstm" className="w-1/6" passHref>
                        <Button size='large' type='primary' className="w-full h-full shadow shadow-white" ><p className="text-white">ENSEMBLE CNN + BILSTM</p></Button>
                    </Link>
                    <Link href="/show-model/phobert/fusion_cnn_bilstm" className="w-1/6" passHref>
                        <Button size='large' type='primary' className="w-full h-full shadow shadow-white" ><p className="text-white">FUSION CNN + BILSTM</p></Button>
                    </Link>
                </Row>
            </Col>
        </Col >
    )
}
'use client'
import { Col, Row, Space, Card, Button, Typography } from "antd"


const { Title, Text } = Typography
export default function ModelSelectPage() {
    return (
        <Col className="w-full h-full flex flex-col justify-center items-center align-middle">
            <Col className="w-full h-1/2 flex flex-col justify-center items-center">
                <Title><p className="text-white text-center">Global Vector Embedding</p></Title>
                <Row className="w-full h-32 gap-5" justify={'center'}>
                    <Button size='large' type='primary' className="w-1/6 h-full shadow shadow-white" ><p className="text-white">LSTM MODEL</p></Button>
                    <Button size='large' type='primary' className="w-1/6 h-full shadow shadow-white" ><p className="text-white">CNN MODEL</p></Button>
                    <Button size='large' type='primary' className="w-1/6 h-full shadow shadow-white" ><p className="text-white">BiLSTM MODEL</p></Button>
                    <Button size='large' type='primary' className="w-1/6 h-full shadow shadow-white" ><p className="text-white">ENSEMBLE CNN + BILSTM</p></Button>
                    <Button size='large' type='primary' className="w-1/6 h-full shadow shadow-white" ><p className="text-white">FUSION CNN + BILSTM</p></Button>
                </Row>
            </Col>
            <Col className="w-full h-1/2 flex flex-col justify-center items-center">
                <Title><p className="text-white text-center">PhoBERT Embedding</p></Title>
                <Row className="w-full h-32 gap-5" justify={'center'}>
                    <Button size='large' type='primary' className="w-1/6 h-full shadow shadow-white" ><p className="text-white">LSTM MODEL</p></Button>
                    <Button size='large' type='primary' className="w-1/6 h-full shadow shadow-white" ><p className="text-white">CNN MODEL</p></Button>
                    <Button size='large' type='primary' className="w-1/6 h-full shadow shadow-white" ><p className="text-white">BiLSTM MODEL</p></Button>
                    <Button size='large' type='primary' className="w-1/6 h-full shadow shadow-white" ><p className="text-white">ENSEMBLE CNN + BILSTM</p></Button>
                    <Button size='large' type='primary' className="w-1/6 h-full shadow shadow-white" ><p className="text-white">FUSION CNN + BILSTM</p></Button>
                </Row>
            </Col>
        </Col>
    )
}
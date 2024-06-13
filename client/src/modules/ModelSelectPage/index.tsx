'use client'
import { Col, Row, Space, Card, Button, Typography, Layout } from "antd"
import Link from 'next/link';
import { GLOVE_INFO, PHOBERT_INFO, MODEL_LIST } from '@/constants/constants'

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
    )
}

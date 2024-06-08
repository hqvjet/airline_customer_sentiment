'use client'
import { Col, Row, Form, Input, Button, Space, Card, Typography, FormProps, message } from "antd"
import { MdOutlineTitle } from "react-icons/md"
import { usePathname } from 'next/navigation'
import useAPI from "@/apis"
import { useState } from "react"
import { getSentiment } from "@/constants/utils"

const { Title, Text } = Typography
const { TextArea } = Input

type Comment = {
    title: string
    content: string
}

export default function ShowModelPage() {
    const path = usePathname()
    const [prediction, setPrediction] = useState<Array<number>>([])
    const [sentiment, setSentiment] = useState<string>('')

    const onFinishForm: FormProps<Comment>['onFinish'] = (values: any) => {
        const apiFunction = useAPI(path);

        if (apiFunction !== null) {
            apiFunction(values.title, values.content)
                .then((response: any) => {
                    let array = []
                    for (var i = 0; i < response.data.prediction[0].length; i += 1) {
                        array.push(response.data.prediction[0][i].toFixed(4))
                    }
                    if (array.length == 0)
                        array.push(response.data.prediction[0])
                    setPrediction(array)
                    setSentiment(getSentiment(array))
                    message.success('Post Successful!')
                })
                .catch((error: any) => {
                    console.log(error)
                })
        } else {
            message.error("URL error");
        }
    }

    const onFinishFailed: FormProps<Comment>["onFinishFailed"] = (errorInfo) => {
        message.error('Please check your input!')
    }

    return (
        <Space direction="vertical" className="">
            <Card className="bg-transparent p-10">
                <Col>
                    <Title><p className="text-white">LEAVE A COMMENT</p></Title>
                    <Form
                        name='sign-in'
                        onFinish={onFinishForm}
                        onFinishFailed={onFinishFailed}
                    >
                        <Form.Item
                            name='title'
                            rules={[
                                {
                                    required: true,
                                    message: 'Please leave your title here!',
                                },
                            ]}
                        >
                            <Input prefix={<MdOutlineTitle />} placeholder="Leave Title Here" autoComplete="off" />
                        </Form.Item>
                        <Form.Item
                            name='content'
                            rules={[
                                {
                                    required: true,
                                    message: 'Please leave your content here!',
                                },
                                {
                                    max: 200,
                                    message: 'Max content\'s length is 200 charaters'
                                }
                            ]}
                        >
                            <TextArea placeholder="Leave your content here" maxLength={200} rows={4} />
                        </Form.Item>
                        <Form.Item>
                            <Button type="primary" htmlType="submit" className="bg-green-500 w-full">
                                Post
                            </Button>
                        </Form.Item>
                    </Form>
                    <Row justify={'center'} align={'middle'}>
                        <Title><p className={`${sentiment == 'Negative' ? 'text-red-500' : sentiment == 'Neutral' ? 'text-yellow-300' : 'text-green-400'}`}>{sentiment}</p></Title>
                    </Row>
                </Col>
            </Card>
            {prediction.length > 1 ?
            (<Row justify={'center'} align={'middle'}>
                <Col className="w-2/3">
                    <Title level={3}><p className="text-white">Negative: {prediction[0]}</p></Title>
                    <Title level={3}><p className="text-white">Neutral: {prediction[1]}</p></Title>
                    <Title level={3}><p className="text-white">Positive: {prediction[2]}</p></Title>
                </Col>
            </Row>) :
            (<></>)
            }
        </Space>
    )
}

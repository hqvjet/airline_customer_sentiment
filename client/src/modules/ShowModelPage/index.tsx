'use client'
import { Col, Form, Input, Button, Space, Card, Typography, FormProps, message } from "antd"
import { MdOutlineTitle } from "react-icons/md"
import { usePathname } from 'next/navigation'
import useAPI from "@/apis"

const { Title, Text } = Typography
const { TextArea } = Input

type Comment = {
    title: string
    content: string
}

export default function ShowModelPage() {
    const path = usePathname()

    const onFinishForm: FormProps<Comment>['onFinish'] = (values: any) => {
        const apiFunction = useAPI(path);
        console.log(values)

        if (apiFunction !== null) {
            apiFunction(values.title, values.content)
                .then((response: any) => {
                    console.log(response)
                })
                .catch((error: any) => {
                    console.log(error)
                });
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
                            <Input prefix={<MdOutlineTitle />} placeholder="Leave Title Here" />
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
                </Col>
            </Card>
        </Space>
    )
}
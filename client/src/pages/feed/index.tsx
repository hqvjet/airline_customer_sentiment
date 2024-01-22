import React, { useEffect, useState } from "react"
import Comment from "../api/comment"
import { Col, Row, Typography, Button, Space, Image } from "antd"
import { AirlineType } from "@/constants/DataType"
import { FaArrowRight, FaArrowLeft } from 'react-icons/fa'
const { Title, Text } = Typography

//MOCK DATA
import { AIRLINE } from './data'

export default function Feed() {


    const submit = () => {
        const title = document.getElementById('title') as HTMLInputElement
        const comment = document.getElementById('cmt') as HTMLInputElement
        Comment.postComment(title?.value, comment?.value)
            .then(r => r.data.rating)
    }

    return (
        <Row>
            <Row className="w-full">
            </Row>
            <Space direction="vertical" className="my-20 mx-7">
                <Row justify={'center'} align={'middle'} className="my-1 w-full">
                    <Col md={12} sm={24} xs={24}>
                        <Row justify={'center'}>
                            <Col md={20} sm={24} xs={24}>
                                <Image src={'/images/bee.png'} preview={false} className="object-cover rounded" />
                            </Col>
                        </Row>
                    </Col>
                    <Col md={12}>
                        <Space direction="vertical">
                            <Title>AI-FIRST AGRICULTURAL CROWDFUNDING PLATFORM</Title>
                            <Text className="text-left">
                                A dedicated crowdfunding platform for farmers directly addresses food loss by
                                aligning production with actual market demand, ensuring financial stability, and
                                fostering investment in sustainable, waste-reducing technologies.
                            </Text>
                            <Button size="large" type="primary">
                                Learn more
                            </Button>
                        </Space>
                    </Col>
                </Row>
                <Row className="my-20" justify={'space-between'}>
                    <Row justify={'center'} className="w-full mb-20">
                        <Title>
                            What Challenges is{' '}
                            <span className="self-center font-extrabold text-4xl text-third whitespace-nowrap">
                                {' '}
                                Econations
                            </span>{' '}
                            Addressing
                        </Title>
                    </Row>
                    <Row justify={'space-between'} className="w-full">
                        <Col md={6} sm={6} xs={24} className="mb-10">
                            <Space direction="vertical" align="center" className="flex justify-center">
                                <Image
                                    src={'/images/wasted.jpg'}
                                    preview={false}
                                    width={'20em'}
                                    height={'15em'}
                                    className="object-cover rounded"
                                />
                                <Text className="text-center flex items-center">
                                    Farm-stage waste due to overproduction
                                </Text>
                            </Space>
                        </Col>
                        <Col md={6} sm={6} xs={24} className="mb-10">
                            <Space direction="vertical" align="center" className="flex justify-center">
                                <Image
                                    src={'/images/food.png'}
                                    preview={false}
                                    width={'20em'}
                                    height={'15em'}
                                    className="object-cover rounded"
                                />
                                <Text className="text-center flex items-center">
                                    Inefficiencies in the current supply chain
                                </Text>
                            </Space>
                        </Col>
                        <Col md={6} sm={6} xs={24} className="mb-10">
                            <Space direction="vertical" align="center" className="flex justify-center">
                                <Image
                                    src={'/images/instable.jpg'}
                                    preview={false}
                                    width={'20em'}
                                    height={'15em'}
                                    className="object-cover rounded"
                                />
                                <Text className="text-center flex items-center">
                                    Financial Stability for Farmers
                                </Text>
                            </Space>
                        </Col>
                    </Row>
                </Row>
                <Row className="my-20 w-full" justify={'center'}>
                    <Space direction="vertical">
                        <Row justify={'center'}>
                            <Title>Our solution</Title>
                        </Row>
                        <Row justify={'center'}>
                            <Text>
                                Building a Platform that empowers individual farms to establish their own
                                crowdfunding channels and reporting systems
                            </Text>
                        </Row>
                        <Row justify={'space-between'} className="my-20">
                            <Col sm={24} md={7} xs={24} className="relative mb-32">
                                <Image src="/images/ong-mat-6025.jpg" preview={false} className="rounded" />
                                <div className="p-4 bg-slate-100 border border-gray-200 rounded-lg hover:shadow-lg sm:p-8 absolute -bottom-12 inset-x-5">
                                    <div className="flow-root text-center">
                                        <Space direction="vertical">
                                            <Text>Honey Bee</Text>
                                            <span className="text-primary flex items-center justify-between cursor-pointer">
                                                Readmore <FaArrowRight className="ml-3" />
                                            </span>
                                        </Space>
                                    </div>
                                </div>
                            </Col>
                            <Col sm={24} md={7} xs={24} className="relative mb-32">
                                <Image src="/images/mango.jpg" preview={false} className="rounded" />
                                <div className="p-4 bg-slate-100 border border-gray-200 rounded-lg hover:shadow-lg sm:p-8 absolute -bottom-12 inset-x-5">
                                    <div className="flow-root text-center">
                                        <Space direction="vertical">
                                            <Text>Mango Tree</Text>
                                            <span className="text-primary flex items-center justify-between cursor-pointer">
                                                Readmore <FaArrowRight className="ml-3" />
                                            </span>
                                        </Space>
                                    </div>
                                </div>
                            </Col>
                            <Col sm={24} md={7} xs={24} className="relative mb-32">
                                <Image src="/images/stingless-bee.png" preview={false} className="rounded" />
                                <div className="p-4 bg-slate-100 border border-gray-200 rounded-lg hover:shadow-lg sm:p-8 absolute -bottom-12 inset-x-5">
                                    <div className="flow-root text-center">
                                        <Space direction="vertical">
                                            <Text>Stingless bee</Text>
                                            <span className="text-primary flex items-center justify-between cursor-pointer">
                                                Readmore <FaArrowRight className="ml-3" />
                                            </span>
                                        </Space>
                                    </div>
                                </div>
                            </Col>
                        </Row>
                    </Space>
                </Row>
                <Row className="my-20">
                    <Row justify={'center'} className="w-full">
                        <Title className="text-center">Our Partners</Title>
                    </Row>
                    <Row justify={'center'} className="w-full" align={'bottom'}>
                        <Col md={6} xs={24}>
                            <Row justify={'center'}>
                                <Image src="/icons/trung-tam-nghi-luc-song.png" preview={false} />
                            </Row>
                        </Col>
                        <Col md={6} xs={24}>
                            <Row justify={'center'}>
                                <Image src="/icons/ecovi-logo.png" preview={false} />
                            </Row>
                        </Col>
                        <Col md={6} xs={24}>
                            <Row justify={'center'}>
                                <Image src="/icons/treebank-logo.png" preview={false} />
                            </Row>
                        </Col>
                    </Row>
                </Row>
                <Row className="my-20">
                    <Row justify={'center'} className="w-full">
                        <Title className="text-center">Our happy clients</Title>
                    </Row>
                    <Row justify={'space-between'} className="w-full" align={'middle'}>
                        <Col md={2} sm={0} xs={0}>
                            <Space className="p-3 rounded-full bg-slate-100 cursor-pointer">
                                <FaArrowLeft className="text-black" />
                            </Space>
                        </Col>
                        <Col md={8} sm={0} xs={0}>
                            <div className="p-4 bg-slate-100 border border-gray-200 rounded-lg shadow-lg sm:p-8">
                                <div className="flow-root text-center">
                                    <Row>
                                        <Row className="my-2">
                                            <Text className="text-left line-clamp-6">
                                                Econations truly stands out as a beacon of quality and sustainability in the
                                                realm of honey and mango sales. The commitment to offering pure,
                                                unadulterated honey reflects a dedication to providing customers with a
                                                natural and wholesome product. The rich flavor profile of the honey and the
                                                succulent sweetness of the mangoes are a testament to the careful
                                                cultivation and harvesting processes employed by Econations. This project
                                                not only promotes healthy living but also supports local agriculture, making
                                                it a commendable venture that values both consumers and the environment
                                            </Text>
                                        </Row>
                                        <Row className="my-2 w-full" align={'middle'}>
                                            <Col md={5}>
                                                <Image
                                                    src="/images/malefarmer.png"
                                                    preview={false}
                                                    className="rounded-full object-cover"
                                                    width={'3em'}
                                                    height={'3em'}
                                                />
                                            </Col>
                                            <Col md={18} className="ml-2">
                                                <Row>
                                                    <Text className="font-bold">Hoang Nguyen</Text>
                                                </Row>
                                                <Row>
                                                    <Text className="text-slate-400">Customer</Text>
                                                </Row>
                                            </Col>
                                        </Row>
                                    </Row>
                                </div>
                            </div>
                        </Col>
                        <Col md={8}>
                            <div className="p-4 bg-slate-100 border border-gray-200 rounded-lg shadow-lg sm:p-8">
                                <div className="flow-root text-center">
                                    <Row>
                                        <Row className="my-2">
                                            <Text className="text-left line-clamp-6">
                                                Econations is a refreshing initiative that seamlessly blends the goodness of
                                                honey and the delight of mangoes. The attention to detail in ensuring the
                                                highest quality of both products is evident in every jar of honey and every
                                                piece of mango offered. The emphasis on eco-friendly practices further
                                                elevates the project, aligning with the growing global awareness of
                                                sustainable living. By choosing Econations, consumers not only get to enjoy
                                                premium honey and mangoes but also contribute to a positive environmental
                                                impact. It's a win-win for those seeking top-notch products with a
                                                conscience.
                                            </Text>
                                        </Row>
                                        <Row className="my-2 w-full" align={'middle'}>
                                            <Col md={5}>
                                                <Image
                                                    src="/images/femalefarmer.png"
                                                    preview={false}
                                                    className="rounded-full object-cover"
                                                    width={'3em'}
                                                    height={'3em'}
                                                />
                                            </Col>
                                            <Col md={18} className="ml-2">
                                                <Row>
                                                    <Text className="font-bold">Lin Nguyen</Text>
                                                </Row>
                                                <Row>
                                                    <Text className="text-slate-400">Customer</Text>
                                                </Row>
                                            </Col>
                                        </Row>
                                    </Row>
                                </div>
                            </div>
                        </Col>
                        <Col md={2} sm={0} xs={0}>
                            <Space className="flex justify-end">
                                <Space className="p-3 rounded-full bg-slate-100 cursor-pointer">
                                    <FaArrowRight className="text-black " />
                                </Space>
                            </Space>
                        </Col>
                    </Row>
                </Row>
            </Space>
        </Row>
    )
}

import { Row, Col, Typography, Input, Button, Image, Card } from 'antd'
import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import usingAirline from '../api/AirlineAPI'
import usingComment from '../api/CommentAPI'
import { getRating } from '../helper/utils'

const { Title, Text } = Typography

export default function AirlineDetail() {

    const { id } = useParams()
    const [airline, setAirline] = useState({
        id: 0,
        name: '',
        about: '',
        rating: 0
    })
    const [comments, setComments] = useState([])
    const [img, setImg] = useState()
    useEffect(() => {
        if (id != undefined || id != null) {
            usingAirline.getAirline(id)
                .then(res => setAirline(res.data[0]))
                .catch(err => console.log(err))

            usingComment.getComments(id)
                .then(res => {
                    setComments(res.data)
                    console.log(res.data)
                })
                .catch(err => console.log(err))

            usingAirline.getThumbnail(id)
                .then(res => setImg(URL.createObjectURL(res.data)))

        }
    }, [id])

    const submitComment = () => {
        if (id != undefined || id != null) {
            const title = document.getElementById('title_input').value
            const comment = document.getElementById('comment_input').value

            usingComment.submitComment(id, title, comment)
                .then(() => {
                    usingComment.getComments(id)
                        .then(res => {
                            setComments(res.data)
                            usingAirline.getAirline(id)
                                .then(res => setAirline(res.data[0]))
                                .catch(err => console.log(err))
                        })
                        .catch(err => console.log(err))
                })
                .catch(err => console.log(err))
        }
    }

    return (
        <Col md={24}>
            <Row className='flex'>
                <Col className='p-5 flex-1' md={8}>
                    <Image
                        src={img}
                        className='object-contain rounded-md'
                    />
                </Col>
                <Col className='p-5 flex-1' md={16}>
                    <Card>
                        <Row className='gap-10'>
                            <Col md={20}>
                                <Title>{airline.name}</Title>
                                <Text>{airline.about}</Text>
                            </Col>
                            <Col className="flex justify-center items-center" md={1}>
                                <Col className="px-5 py-8 bg-gradient-to-b from-green-500 via-orange-400 to-red-500">
                                    <Title level={4}>{airline.rating}</Title>
                                </Col>
                            </Col>
                        </Row>
                    </Card>
                </Col>
            </Row>
            <Col md={20}>
                <Row className='p-5' justify={'space-between'}>
                    <Col md={9}>
                        <Input placeholder='hello' className="flex-grow border-b-2 shadow-md focus:ring-2 focus:ring-blue-500" id='title_input' />
                    </Col>
                    <Col md={9}>
                        <Input placeholder='hello' className='p-2' id='comment_input' />
                    </Col>
                    <Col md={2}>
                        <Button
                            type='primary'
                            className='text-black'
                            onClick={submitComment}
                        >Submit</Button>
                    </Col>
                </Row>
                <Col className='p-5'>
                    {comments.length === 0 && (
                        <Col align={'middle'}>
                            <Title level={3}>Hãng bay này chưa được đánh giá</Title>
                        </Col>
                    )}
                    {comments.map(comment => (
                        <Col key={comment.id} className='mb-5'>
                            <Card
                                title={
                                    <div className='flex gap-5'>
                                        {comment.title}
                                        <Text className={`${comment.rating === 'pos' ? 'text-green-600' :
                                            (comment.rating === 'neu' ? 'text-yellow-600' :
                                                'text-red-600')}`}>
                                            {comment.rating}
                                        </Text>
                                    </div>
                                }
                                type='inner'
                                className='shadow'
                            >
                                <Row>{comment.comment}</Row>
                            </Card>

                        </Col>
                    ))}
                </Col>
            </Col>
        </Col>
    )
}
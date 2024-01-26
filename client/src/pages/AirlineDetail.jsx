import { Row, Col, Typography, Input, Button } from 'antd'
import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { Airlines } from '../data/AirLineData'
import { AirLineComments } from '../data/AirLineComment'
import usingAirline from '../api/AirlineAPI'
import usingComment from '../api/CommentAPI'

const { Title, Text } = Typography

export default function AirlineDetail() {

    const { id } = useParams()
    const [airline, setAirline] = useState({
        id: 0,
        name: ''
    })
    const [comments, setComments] = useState([])
    useEffect(() => {
        if (id != undefined || id != null) {
            usingAirline.getAirline(id)
                .then(res => setAirline(res.data))
                .catch(err => console.log(err))

            usingComment.getComments(id)
                .then(res => {
                    setComments(res.data)
                    console.log(res.data)
                })
                .catch(err => console.log(err))
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
                            console.log(res.data)
                        })
                        .catch(err => console.log(err))
                })
                .catch(err => console.log(err))
        }
    }

    return (
        <Col>
            <Row>
                <Col>IMAGE</Col>
                <Col>Information</Col>
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
                    {comments.map(comment => (
                        <Col key={comment.id} className='mb-10'>
                            <Row justify={'start'} align={'middle'} className='gap-5'>
                                <Title level={4}>{comment.title}</Title>
                                <Text
                                    className={`${comment.rating === 'pos' ? 'text-green-600' :
                                        (comment.rating === 'neu' ? 'text-yellow-600' :
                                            'text-red-600')}`}
                                >{comment.rating}</Text>
                            </Row>
                            <Row>{comment.comment}</Row>
                        </Col>
                    ))}
                </Col>
            </Col>
        </Col>
    )
}
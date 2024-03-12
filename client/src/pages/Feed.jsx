import { Row, Col, Typography, Card } from 'antd'
import { Airlines } from '../data/AirLineData'
import { useEffect, useState } from 'react'
import usingAirline from '../api/AirlineAPI'
import { useParams } from 'react-router-dom'

const { Title, Text } = Typography

export default function Feed() {
    const { id } = useParams()
    const [airlines, setAirlines] = useState([])
    useEffect(() => {
        usingAirline.getAirlines(id)
            .then(res => {
                setAirlines(res.data)
            })
            .catch(err => console.log(err))
    }, [])

    const truncateContent = (content) => {
        if (content.length <= 150) {
            return content;
        } else {
            return content.substring(0, 150) + '...'
        }
    }

    return (
        <Row className='mx-10'>
            <Row justify={'center'} className='w-full my-10'>
                <Title>AIRLINE FILTER</Title>
            </Row>
            <Row className='w-full'>
                {airlines.map(airline => (
                    <Col key={airline.id} md={7} className='m-5'>
                        <Card>
                            <Row className='' justify={'center'}>
                                <Title className='text-black'><a href={`/airline/detail/${airline.id}`}>{airline.name}</a></Title>
                            </Row>
                            <Text>
                                {truncateContent(airline.about)}
                            </Text>
                        </Card>
                    </Col>
                ))}
            </Row>
        </Row>
    )
}
import { Row, Col, Typography } from 'antd'
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
                console.log(res.data)
                setAirlines(res.data)
            })
            .catch(err => console.log(err))
    }, [])

    return (
        <Row className='mx-10'>
            <Row justify={'center'} className='w-full my-10'>
                <Title>AILRINE FILTER</Title>
            </Row>
            <Row gutter={[16, 24]}>
                {airlines.map(airline => (
                    <Col key={airline.id} span={8} className='gutter-row'>
                        <Row className='shadow p-10' justify={'center'}>
                            <a href={`/airline/detail/${airline.id}`}>{airline.name}</a>
                        </Row>
                    </Col>
                ))}
            </Row>
        </Row>
    )
}
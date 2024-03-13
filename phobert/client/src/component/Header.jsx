import { navbar } from "../constant/constants"
import { Col, Row, Breadcrumb, Typography } from 'antd'

const {Title, Text} = Typography

export default function Header() {
    return (
        <Col className="flex justify-between p-2 bg-gray-300">
            <Col>
                LOGO
            </Col>
            <Row justify={'center'} className="gap-10">
                {navbar.map((item, index) => (
                    <Row key={index}>
                        <a href={item.href} className="hover:text-gray-400">
                            <Row className="flex items-center justify-center gap-2">
                                <item.icon size={15}/>
                                <div className="text-xl font-bold">{item.name}</div>
                            </Row>
                        </a>
                    </Row>
                ))}
            </Row>
        </Col>
    )
}
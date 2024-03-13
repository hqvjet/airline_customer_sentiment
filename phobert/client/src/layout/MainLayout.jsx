import { Outlet, Link } from "react-router-dom";
import { Row, Col, Space } from 'antd'
import Header from '../component/Header'

export default function MainLayout({ children }) {
    return (
        <Col>
            <Header/>
            <Outlet/>
        </Col>
    )
}
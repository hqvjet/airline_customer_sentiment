import { BrowserRouter, Routes, Route } from "react-router-dom"
import MainLayout from "./layout/MainLayout"
import Home from "./pages/Home"
import Feed from "./pages/Feed"
import AirlineDetail from "./pages/AirlineDetail"

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<MainLayout />}> 
          <Route index element={<Home />} />
          <Route path="feed" element={<Feed />} />
          <Route path="airline/detail/:id" element={<AirlineDetail />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
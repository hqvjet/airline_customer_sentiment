import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { Col, Row, Space } from "antd";

const inter = Inter({ subsets: ["latin"] });

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={inter.className}>
          <Row align={"middle"} justify={"center"} className="w-full h-screen">
            {children}
          </Row>
      </body>
    </html>
  );
}

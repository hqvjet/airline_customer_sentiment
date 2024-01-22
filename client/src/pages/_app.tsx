import '@/styles/globals.css'
import type { AppProps } from 'next/app'
import { ConfigProvider } from 'antd';
import MainLayout from '@/layout/MainLayout'

import theme from '@/theme/themeConfig';

export default function App({ Component, pageProps }: AppProps) {
  return (
    <ConfigProvider theme={theme}>
      <MainLayout>
        <Component {...pageProps} />
      </MainLayout>
    </ConfigProvider>
  )
}
import axios from 'axios'

export const AIRLINE_CONFIG = axios.create({
    baseURL: process.env.REACT_APP_API_ENDPOINT + '/airlines',
    timeout: 10000
})

export const COMMENT_CONFIG = axios.create({
    baseURL: process.env.REACT_APP_API_ENDPOINT + '/comments',
    timeout: 10000
})
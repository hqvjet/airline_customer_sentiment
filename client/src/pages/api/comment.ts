import axios from 'axios'

const Comment_Config = axios.create({
    baseURL: 'http://localhost:8080',
})

const Comment = {
    postComment: async (title: string, comment: string) => {
        return await Comment_Config.post('/', { title: title, text: comment})
    },
}

export default Comment
import { COMMENT_CONFIG } from '.'

const usingComment = {
    getComments: (id) => {
        return COMMENT_CONFIG.get(`/get_comments/${id}`)
    },
    submitComment: (id, title, comment) => {
        return COMMENT_CONFIG.post(`/submit_comment/${id}`, {
            title: title,
            comment: comment
        })
    }
}

export default usingComment
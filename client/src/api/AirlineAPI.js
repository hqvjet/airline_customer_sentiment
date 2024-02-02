import { AIRLINE_CONFIG } from '.'

const usingAirline = {
    getAirline: (id) => {
        return AIRLINE_CONFIG.get(`/get_airline/${id}`)
    },
    getAirlines: () => {
        return AIRLINE_CONFIG.get('/get_airlines')
    },
    getAirlinesRating: (id) => {
        return AIRLINE_CONFIG.get(`/rate_airline/${id}`)
    },
    getThumbnail: (id) => {
        return AIRLINE_CONFIG.get(`/get_thumbnail/${id}`, {responseType: 'blob'})
    }
}

export default usingAirline
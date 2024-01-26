import { AIRLINE_CONFIG } from '.'

const usingAirline = {
    getAirline: (id) => {
        return AIRLINE_CONFIG.get(`/get_airline/${id}`)
    },
    getAirlines: () => {
        return AIRLINE_CONFIG.get('/get_airlines')
    }
}

export default usingAirline
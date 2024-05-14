export function getSentiment(array: Array<number>): string {
    console.log(array)
    if (array.length == 1) {
        if (array[0] == 0)
            return 'Negative'
        if (array[0] == 1)
            return 'Neural'
        return 'Positive'
    }
    var max = array[0]
    var index = 0

    for(var i = 1; i < array.length; i += 1) {
        if (max < array[i]) {
            max = array[i]
            index = i
        }
    }

    if (index == 0) 
        return 'Negative'
    else if (index == 1)
        return 'Neutral'
    else    
        return 'Positive'
}

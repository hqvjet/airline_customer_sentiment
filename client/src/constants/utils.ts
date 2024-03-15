export function getSentiment(array: Array<number>): string {
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
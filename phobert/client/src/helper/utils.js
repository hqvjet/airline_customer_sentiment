export function getRating(rating) {
    return ((rating.pos + rating.neu + rating.neg) / 3).toFixed(2)
}
from flask import Blueprint, request, jsonify
import httpx
from database import connect_db
from airline import rate_airline


def generate_id():
    db = connect_db()
    cursor = db.cursor()

    try:
        cursor.execute('SELECT COUNT(`id`) FROM `comment`')
        count = cursor.fetchone()[0] + 1
        count = str(count)
        cursor.close()
        db.close()
    except:
        return jsonify({'error': 'SQL Error'})

    padding = ''
    for _ in range(4 - len(count)):
        padding += '0'

    return 'CM' + padding + count

comments = Blueprint('comments', __name__, url_prefix="/api/v1/comments")

@comments.get('/get_comments/<airline_id>')
def get_airlines(airline_id):
    db = connect_db()
    cursor = db.cursor(dictionary=True)
    cursor.execute('SELECT `id`, `title`, `comment`, `rating` FROM `comment` WHERE airline_id = %s', (airline_id,))
    res = cursor.fetchall()
    cursor.close()
    db.close()

    return jsonify(res)

@comments.post('/submit_comment/<airline_id>')
async def submit_airlines(airline_id):
    db = connect_db()
    cursor = db.cursor(dictionary=True)
    req = request.get_json()

    title = req.get('title')
    comment = req.get('comment')
    id = generate_id()

    # Get rating
    api = 'http://localhost:4001/'
    data = {'title': title, 'text': comment}

    async with httpx.AsyncClient() as Client:
        try:
            res = await Client.post(api, json=data)
            rating = res.json().get('rating')
            print('rating', rating)
        except httpx.RequestError as e:
            return jsonify({'error': f'Error connecting to API: {str(e)}'})

    try:
        cursor.execute('INSERT INTO `comment`(`id`, `airline_id`, `title`, `comment`, `rating`) VALUES (%s, %s, %s, %s, %s)', (id, airline_id, title, comment, rating,))
        db.commit()
        cursor.close()
        db.close()

        rate_airline(airline_id)
    except:
        return jsonify({'error': 'SQL Error'})

    return jsonify({'info': 'Successful submitted'})

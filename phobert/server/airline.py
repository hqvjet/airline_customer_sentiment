from flask import Blueprint, jsonify, send_file
from database import connect_db
import os

airlines = Blueprint('airlines', __name__, url_prefix="/api/v1/airlines")


@airlines.get('/get_airlines')
def get_airlines():
    db = connect_db()
    cursor = db.cursor(dictionary=True)
    cursor.execute('SELECT * FROM `airline`')
    res = cursor.fetchall()
    cursor.close()
    db.close()

    return res

@airlines.get('/get_thumbnail/<id>')
def get_thumbnail(id):
    thumbnail_path = os.path.join(os.getcwd(), 'resources', 'images', f'{id}.jpg')
    if os.path.exists(thumbnail_path):
        return send_file(thumbnail_path, mimetype='image/jpeg')
    else:
        return 'Thumbnail not found', 404

@airlines.get('/get_airline/<id>')
def get_airline(id):
    db = connect_db()
    cursor = db.cursor(dictionary=True)
    cursor.execute('SELECT * FROM `airline` WHERE id=%s', (id,))
    res = cursor.fetchall()
    cursor.close()
    db.close()

    print(res)
    return res

def rate_airline(id):
    db = connect_db()
    cursor = db.cursor()
    cursor.execute('SELECT COUNT(`id`) AS post_count FROM `comment` WHERE `airline_id`=%s AND `rating`=%s', (id, 'pos'))
    pos = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(`id`) AS post_count FROM `comment` WHERE `airline_id`=%s AND `rating`=%s', (id, 'neu'))
    neu = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(`id`) AS post_count FROM `comment` WHERE `airline_id`=%s AND `rating`=%s', (id, 'neg'))
    neg = cursor.fetchone()[0]

    rating = (3*pos + 2*neu + 1*neg) / (pos + neu + neg)
    rating = '{:.2f}'.format(rating)

    cursor.execute('UPDATE `airline` SET `rating` = %s WHERE `id` = %s', (rating, id,))
    db.commit()

    cursor.close()
    db.close()

    return jsonify({'rating': rating}), 200
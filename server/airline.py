from flask import Blueprint, app, request, jsonify
from database import connect_db

airlines = Blueprint('airlines', __name__, url_prefix="/api/v1/airlines")


@airlines.get('/get_airlines')
def get_airlines():
    db = connect_db()
    cursor = db.cursor(dictionary=True)
    cursor.execute('SELECT * FROM `airline`')
    res = cursor.fetchall()
    cursor.close()
    db.close()

    print(res)
    return res


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

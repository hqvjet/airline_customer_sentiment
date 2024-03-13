import mysql.connector


def connect_db():
    try:

        return mysql.connector.connect(
            host='127.0.0.1',
            user='root',
            password='',
            database='airline',
        )

    except mysql.connector.Error as error:
        print("Error while connecting to MySQL", error)
        return None

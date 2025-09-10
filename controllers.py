from flask import request, jsonify
from services import student_file_processing_service


def handle_student_file_processing_controller():
    print("request received in ")
    file = request.files['file']

    try:
        output = student_file_processing_service(file)

        return jsonify(output), 200


    except ValueError as ve:

        # validation error -> 400 Bad Request

        return jsonify({'error': str(ve)}), 400


    except Exception as e:

        # any other server-side error -> 500 Internal Server Error

        return jsonify({'error': str(e)}), 500

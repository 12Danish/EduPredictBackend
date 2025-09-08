from flask import request, jsonify
from services import student_file_processing_service


def handle_student_file_processing_controller():
    print("request received in ")
    file = request.files['file']

    try:
        student_file_processing_service(file)

        return jsonify({"message" : "Received request successfully"}),200

    except Exception as e:
        return jsonify({'error': str(e)}), 404

from flask import jsonify

from controllers import handle_student_file_processing_controller


def register_routes(app):
    app.add_url_rule(
        '/process_file',
        view_func=handle_student_file_processing_controller,
        methods=['POST']
    )

    @app.errorhandler(404)
    def page_not_found(e):
        return jsonify({"message": "resource not found"}), 404

    @app.errorhandler(500)
    def internal_server_error(e):
        return jsonify({"message": "internal server error"}), 500

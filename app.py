from flask import Flask, request, jsonify
from flask_cors import CORS
from rag_chat import GovtSchemeChatbot  # Replace with actual import

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
chatbot = GovtSchemeChatbot()


@app.route("/")
def health_check():
    return jsonify(
        {
            "status": "active",
            "version": "1.0.0",
            "endpoints": [
                {"path": "/api/start", "methods": ["POST"]},
                {"path": "/api/chat", "methods": ["POST"]},
            ],
        }
    )


@app.route("/api/start", methods=["POST"])
def start_session():
    try:
        session_id = chatbot.start_new_session()
        return jsonify({"session_id": session_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/chat", methods=["POST"])
def handle_chat():
    try:
        data = request.get_json()
        response = chatbot.generate_response(
            query=data.get("query", ""), session_id=data.get("session_id")
        )
        return jsonify(
            {"response": response, "session_id": chatbot.active_session.session_id}
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

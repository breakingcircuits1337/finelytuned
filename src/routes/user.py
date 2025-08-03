from flask import Blueprint, jsonify

user_bp = Blueprint('user', __name__)

@user_bp.route("/register", methods=["POST"])
def register():
    return jsonify({"error": "Not implemented"}), 501

@user_bp.route("/login", methods=["POST"])
def login():
    return jsonify({"error": "Not implemented"}), 501
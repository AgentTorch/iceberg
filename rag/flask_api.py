from flask import Flask, request, jsonify
from flask_cors import CORS  # Add CORS support
from wynm_data_scrape.langchain_rag import retrieve_results
from wynm_data_scrape.langchain_rag import test_voyage_api
from langchain_anthropic import ChatAnthropic
from langchain_voyageai import VoyageAIEmbeddings

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/api/hello', methods=['GET'])
def hello():
    return jsonify({"message": "Hello from the API!"})

@app.route('/api/occupations', methods=['POST'])
def occupations():
    return jsonify({"message": "Occupations!"})

@app.route('/api/chat', methods=['POST'])
def chat():
    print("Chat request received")
    data = request.get_json()
    user_message = data.get('message')  # Get the user's message
    response = retrieve_results(user_message)
    print(response)

    # response = f"{user_message}"
    
    return jsonify({"message": response})


@app.route('/api/chat_test', methods=['POST'])
def chat_test():
    print("Chat request received")
    data = request.get_json()
    user_message = data.get('message')  # Get the user's message
    print(user_message+"\nUser Message Recieved.")
    response = test_voyage_api(user_message)
    print(response + ", Response recieved.")

    # response = f"{user_message}"
    
    return jsonify({"message": response})


if __name__ == '__main__':
    app.run(debug=True)

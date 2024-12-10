from flask import Flask, request, jsonify
import openai
from supabase import create_client, Client
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize clients
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
openai.api_key = OPENAI_API_KEY


def extract_message_details(message_text):
    try:
        messages = [
            {
                "role": "system",
                "content": "You are a expert therapist that analyzes messages and extracts the trigger, thought, and response components.",
            },
            {
                "role": "user",
                "content": f"Analyze the following message and extract the trigger, thought, and response:\n\nMessage: {message_text}\n\nFormat the result as JSON: {{'trigger': '', 'thought': '', 'response': ''}} \n\nDont use markdown or any other formatting",
            },
        ]

        response = openai.chat.completions.create(
            model="gpt-4-turbo-preview", messages=messages
        )
        return eval(response.choices[0].message.content.strip())
    except Exception as e:
        print(f"Error extracting details: {e}")
        return None


def get_embedding(text):
    try:
        response = openai.embeddings.create(input=text, model="text-embedding-3-small")
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding for text '{text}': {e}")
        return None


def process_new_message(new_message):
    try:
        message_id = new_message["id"]
        message_text = new_message["text_message"]

        details = extract_message_details(message_text)
        if not details:
            return {"error": "Failed to extract details"}

        trigger = details["trigger"]
        thought = details["thought"]
        response = details["response"]

        trigger_embedding = get_embedding(trigger)
        thought_embedding = get_embedding(thought)
        response_embedding = get_embedding(response)

        if not all([trigger_embedding, thought_embedding, response_embedding]):
            return {"error": "Failed to retrieve embeddings"}

        result = {
            "message_id": message_id,
            "trigger": trigger,
            "thought": thought,
            "response": response,
            "trigger_embedding": trigger_embedding,
            "thought_embedding": thought_embedding,
            "response_embedding": response_embedding,
        }

        supabase.table("processed_messages").insert(result).execute()
        return {"status": "success", "data": result}
    except Exception as e:
        return {"error": str(e)}


@app.route("/")
def home():
    return jsonify({"status": "healthy", "message": "API is running"})


@app.route("/process-message", methods=["POST"])
def api_process_message():
    try:
        data = request.json
        if not data or "text_message" not in data:
            return jsonify({"error": "Missing text_message in request body"}), 400

        new_message = {
            "id": data.get("id", str(uuid.uuid4())),
            "text_message": data["text_message"],
        }

        result = process_new_message(new_message)
        if "error" in result:
            return jsonify({"error": result["error"]}), 500

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))

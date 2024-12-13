from flask import Flask, request, jsonify
import openai
from supabase import create_client, Client
import os
from dotenv import load_dotenv
from datetime import datetime

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


def evaluate_message_for_assessment(message_text, assessment_type):
    """Evaluate a message for positive/negative outcome based on assessment type."""
    try:
        messages = [
            {
                "role": "system",
                "content": "You are an expert therapist that analyzes messages to determine positive or negative outcomes.",
            },
            {
                "role": "user",
                "content": f"Analyze the following message and determine if it indicates a positive outcome for {assessment_type}. "
                f"Respond with only 'true' for positive outcome or 'false' for negative outcome.\n\nMessage: {message_text}",
            },
        ]

        response = openai.chat.completions.create(
            model="gpt-4o-mini", messages=messages
        )
        return response.choices[0].message.content.strip().lower() == "true"
    except Exception as e:
        print(f"Error evaluating message: {e}")
        return None


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
            "id": str(data.get("id")),
            "text_message": data["text_message"],
        }

        result = process_new_message(new_message)
        if "error" in result:
            return jsonify({"error": result["error"]}), 500

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/assess-message", methods=["POST"])
def assess_message():
    try:
        data = request.json
        if not data or "text_message" not in data or "client_id" not in data:
            return jsonify({"error": "Missing required fields in request body"}), 400

        message_text = data["text_message"]
        client_id = data["client_id"]

        # Get goals and coping mechanisms for the client
        goals = (
            supabase.table("goals")
            .select("client_id, goal")
            .eq("client_id", client_id)
            .execute()
        )
        coping = (
            supabase.table("coping_mechanisms")
            .select("client_id, mechanism")
            .eq("client_id", client_id)
            .execute()
        )

        current_date = datetime.now()

        # Process goals
        for goal in goals.data:
            outcome = evaluate_message_for_assessment(
                message_text, f"goal: {goal['goal']}"
            )
            if outcome is not None:
                supabase.table("assessments").insert(
                    {
                        "client_id": client_id,
                        "assessment": goal["goal"],
                        "outcome": outcome,
                        "assessment_date": current_date.isoformat(),
                    }
                ).execute()

        # Process coping mechanisms
        for mechanism in coping.data:
            outcome = evaluate_message_for_assessment(
                message_text, f"coping mechanism: {mechanism['mechanism']}"
            )
            if outcome is not None:
                supabase.table("assessments").insert(
                    {
                        "client_id": client_id,
                        "assessment": mechanism["mechanism"],
                        "outcome": outcome,
                        "assessment_date": current_date.isoformat(),
                    }
                ).execute()
        general_assessments = ["Positive self-talk", "good mood"]
        for assessment in general_assessments:
            outcome = evaluate_message_for_assessment(message_text, assessment)
            if outcome is not None:
                supabase.table("assessments").insert(
                    {
                        "client_id": client_id,
                        "assessment": assessment,
                        "outcome": outcome,
                        "assessment_date": current_date.isoformat(),
                    }
                ).execute()

        return jsonify({"status": "success", "data": "assessments added"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

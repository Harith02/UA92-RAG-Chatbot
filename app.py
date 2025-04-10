from flask import Flask, render_template, request, session
from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI
import speech_recognition as sr
from langchain.embeddings.openai import OpenAIEmbeddings
from rag import create_rag_chain, get_response
from dotenv import load_dotenv
import os
import markdown

# Load API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Flask app
app = Flask(__name__)
app.secret_key = "your-unique-and-secret-key-here"

# Initialize RAG on startup
CHROMA_DB_DIR = "./ua92_embeddings"  # Updated to new database name
if os.path.exists(CHROMA_DB_DIR):
    # Load existing ChromaDB with the updated name
    vector_store = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=openai_api_key))
    print("Loaded existing ChromaDB (ua92_embeddings)")
else:
    print("Failed to create vector store.")

qa_chain = create_rag_chain(vector_store)
print("RAG chain initialized")

# Speech-to-text function
def speech_to_text():
    recognizer = sr.Recognizer()

    # Use the microphone as the audio source
    with sr.Microphone() as source:
        print("Listening for speech...")
        recognizer.adjust_for_ambient_noise(source)  # Adjust for ambient noise
        audio = recognizer.listen(source)  # Listen to the speech

    try:
        # Recognize speech using Google Web API
        print("Recognizing speech...")
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
        return text
    except sr.UnknownValueError:
        print("Sorry, I could not understand the audio.")
        return None
    except sr.RequestError:
        print("Could not request results from the speech service.")
        return None

@app.route("/", methods=["GET", "POST"])
def index():
    query = ""
    response = ""

    if request.method == "POST":
        if request.form.get("speech_input"):
            query = speech_to_text()
        else:
            query = request.form["query"]

        if query:
            if qa_chain is None:
                response = "Error: RAG not initialized."
            else:
                response = get_response(query, qa_chain)
                response = markdown.markdown(response)
        else:
            response = "Sorry, I couldn't understand the audio."

        return render_template("index.html", query=query, response=response)

    return render_template("index.html", query=query, response=response)

if __name__ == "__main__":
    app.run(debug=False)
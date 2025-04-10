# 🎓 UA92 Admissions Chatbot
A Retrieval-Augmented Generation (RAG) chatbot designed to assist users with University Academy 92 (UA92) admissions-related queries. This project leverages LangChain, GPT-4, and ChromaDB to provide accurate, real-time responses based on scraped university website data.

🚀 Features
🔍 Retrieval-Augmented Generation using GPT-4 and ChromaDB

🗂️ Contextual and semantically accurate responses

🌐 Multilingual support (tested with German, Malay, Chinese, and more)

🎤 Voice input (English only)

📱 Mobile-friendly and accessible interface

📊 97% accuracy in user testing with high satisfaction ratings

🛠 Technologies Used
Python (Flask, LangChain)

ChromaDB (for vector-based document storage and retrieval)

OpenAI GPT-4 API

BeautifulSoup (for web scraping)

JavaScript + HTML/CSS (frontend UI)

SpeechRecognition + Web Speech API (for voice input)

GitHub + Koyeb (for version control and cloud deployment)

📦 Installation
Make sure you have Python 3.8+ installed.

bash
Copy code
git clone https://github.com/yourusername/ua92-chatbot.git
cd ua92-chatbot
pip install -r requirements.txt
Set up your .env file with your OpenAI API key and other credentials:

env
Copy code
OPENAI_API_KEY=your_key_here
Then run the app:

bash
Copy code
python app.py
Open your browser at http://localhost:5000

📄 Dataset
The chatbot was built using content scraped from the UA92 official website, limited to 110 pages to optimise performance and deployment size. The dataset was chunked, cleaned, and embedded into ChromaDB.

🧪 Testing
The chatbot was evaluated using a set of test queries across categories like:

Admissions requirements

Course offerings

Student support services

Achieved:

✅ ~97% response accuracy

✅ Strong semantic understanding

✅ Consistent context retention

✅ User feedback score: 8.71/10

🧠 Known Limitations
❗ Some responses may "hallucinate" when handling rare or ambiguous queries.

📄 Data coverage is static and limited; no real-time updates.

🎤 Voice input currently only supports English.

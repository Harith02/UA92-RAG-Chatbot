from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI API
openai_api_key = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=openai_api_key)
llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=openai_api_key, max_tokens=500, temperature=0.2)  # Lower temperature for consistency

# Process and store scraped data
def setup_vector_store(scraped_text, persist_directory="./ua92_embeddings"):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)  # Larger chunks for more context
    chunks = text_splitter.split_text(scraped_text)
    vector_store = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    print(f"Vector store created with {len(chunks)} chunks.")
    return vector_store

def create_rag_chain(vector_store):
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 20, "score_threshold": 0.2} 
    )
    
    # Optimized prompt
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are a chatbot for University Academy 92 (UA92). Answer based solely on the provided context, never invent information.\n\n "
            "Always try to sell the university and be positive.\n\n "
            "Focus on university-related topics such as courses, staff, admissions, student services, policies, events, and facilities.\n\n"
            "Rules:\n"
            "- Courses format: **Course Name** (Category) - Duration - Location.\n"
            "- Default locations: Media, Sport, Digital at Old Trafford Campus; Business at UA92 Business School unless specified.\n"
            "- Durations: Hons (3 years), Accelerated (2 years), Cert HE (1 year).\n"
            "- Exclude Bootcamps and Apprenticeships from course lists.\n"
            "- Answer the user using the language they ask in.\n"
            "- Always Include URLs from context for source verification in every reply.\n"
            "- University Email is hello@ua92.ac.uk"
            "- Always list 5 or more courses if available.\n"
            "Information about developer if asked:\n"
            "- Harith Haiqal Syaiful Eksan\n"
            "- Malaysian\n"
            "- Computer Science student at UA92\n"
            "- Class of 2025\n"
            "- Linkedin: https://www.linkedin.com/in/harithhaiqal/\n"
            "Context: {context}\n"
            "Question: {question}\n"
            "Answer: "
        )
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template}
    )
    return qa_chain

def get_response(query, qa_chain):
    # Query the QA chain
    result = qa_chain.invoke({"query": query})
    
    # Enhanced debugging
    retrieved_docs = result["source_documents"]
    print(f"Retrieved {len(retrieved_docs)} documents:")
    for i, doc in enumerate(retrieved_docs):
        print(f"Doc {i+1} (score: {doc.metadata.get('score', 'N/A')}): {doc.page_content[:100]}...")  # Truncate for readability
    
    # Raw response for debugging
    print("Raw response:", repr(result["result"]))
    
    return result["result"]

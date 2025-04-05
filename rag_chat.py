import json
import faiss
import numpy as np
import google.generativeai as genai
from config import Config
import uuid
from typing import Dict, List, Optional


class ChatSession:
    """Class to maintain conversation context"""

    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.history: List[Dict] = []
        self.last_schemes: List[Dict] = []


class GovtSchemeChatbot:
    def __init__(self):
        self.config = Config()
        genai.configure(api_key=self.config.GEMINI_API_KEY)
        self.generation_model = genai.GenerativeModel(self.config.GENERATION_MODEL)

        self.schemes = self.load_schemes()
        self.index = self.create_faiss_index()
        self.sessions: Dict[str, ChatSession] = {}
        self.active_session: Optional[ChatSession] = None

    def start_new_session(self) -> str:
        """Initialize and return new session"""
        new_session = ChatSession()
        self.sessions[new_session.session_id] = new_session
        self.active_session = new_session
        return new_session.session_id

    def load_schemes(self) -> List[Dict]:
        """Load and validate schemes data"""
        try:
            with open(self.config.SCHEMES_FILE) as f:
                schemes = json.load(f)

            if not isinstance(schemes, list):
                raise ValueError("Invalid JSON format: Root should be an array")

            return [self.validate_scheme(s) for s in schemes]

        except Exception as e:
            print(f"Error loading schemes: {str(e)}")
            return []

    def validate_scheme(self, scheme: Dict) -> Dict:
        """Validate and normalize scheme format"""
        required_fields = ["title", "description"]
        validated = {
            "text": self.create_scheme_text(scheme),
            "metadata": {
                "title": scheme.get("title", "Untitled Scheme"),
                "description": scheme.get("description", ""),
                "state": scheme.get("state", "National"),
                "eligibility": scheme.get("eligibility", "Contact authorities"),
                "benefits": scheme.get("benefits", []),
                "requirements": scheme.get("requirements", []),
                "application_process": scheme.get("application_process", []),
            },
        }

        for field in required_fields:
            if field not in scheme:
                print(f"Warning: Scheme missing required field '{field}'")

        return validated

    def create_scheme_text(self, scheme: Dict) -> str:
        """Create searchable text from scheme data with validation"""
        fields = {
            "title": ("Title", scheme.get("title", "Untitled Scheme")),
            "description": ("Description", scheme.get("description", "No description")),
            "state": ("State", scheme.get("state", "National")),
            "eligibility": (
                "Eligibility",
                scheme.get("eligibility", "Contact authorities"),
            ),
            "benefits": (
                "Benefits",
                "\n- " + "\n- ".join(scheme.get("benefits", ["No benefits listed"])),
            ),
            "requirements": (
                "Requirements",
                "\n- "
                + "\n- ".join(scheme.get("requirements", ["No requirements listed"])),
            ),
            "application_process": (
                "Application Process",
                "\n- "
                + "\n- ".join(
                    scheme.get("application_process", ["Contact local office"])
                ),
            ),
        }

        return "\n\n".join([f"{name}:\n{value}" for name, value in fields.values()])

    def create_faiss_index(self) -> faiss.IndexFlatL2:
        """Create FAISS index with error handling"""
        try:
            embeddings = self.generate_embeddings()
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings)
            return index
        except Exception as e:
            print(f"Error creating FAISS index: {str(e)}")
            return faiss.IndexFlatL2(768)  # Fallback empty index

    def generate_embeddings(self) -> np.ndarray:
        """Generate embeddings with error handling"""
        embeddings = []
        for scheme in self.schemes:
            try:
                response = genai.embed_content(
                    model=self.config.EMBEDDING_MODEL,
                    content=scheme["text"],
                    task_type="RETRIEVAL_DOCUMENT",
                )
                embeddings.append(response["embedding"])
            except Exception as e:
                print(f"Embedding error: {str(e)}")
                embeddings.append(np.zeros(768))
        return np.array(embeddings)

    def retrieve_schemes(self, query: str, k: int = 3) -> List[Dict]:
        """Safe scheme retrieval with error handling"""
        try:
            query_embedding = genai.embed_content(
                model=self.config.EMBEDDING_MODEL,
                content=query,
                task_type="RETRIEVAL_QUERY",
            )["embedding"]

            _, indices = self.index.search(np.array([query_embedding]), k)
            return [
                self.schemes[i]["metadata"] for i in indices[0] if i < len(self.schemes)
            ]
        except Exception as e:
            print(f"Retrieval error: {str(e)}")
            return []

    def generate_response(self, query: str, session_id: Optional[str] = None) -> str:
        """Generate response with automatic session handling"""
        try:
            # Session management
            if not session_id:
                if not self.active_session:
                    self.start_new_session()
                session = self.active_session
            else:
                session = self.sessions.get(session_id)
                if not session:
                    raise ValueError("Invalid session ID")

            # Retrieve relevant schemes
            session.last_schemes = self.retrieve_schemes(query)

            # Build context-aware prompt
            prompt = self.build_prompt(query, session)

            # Generate response
            response = self.generation_model.generate_content(prompt)
            response_text = response.text

            # Update history
            session.history.append({"user": query, "assistant": response_text})
            session.history = session.history[-3:]  # Keep last 3 exchanges

            return response_text

        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            print(error_msg)
            # Reset session on critical error
            if "session" in locals():
                del self.sessions[session.session_id]
                self.active_session = None
            return error_msg

    def build_prompt(self, query: str, session: ChatSession) -> str:
        """Construct strict context-aware prompt"""
        history_context = (
            "\n".join(
                [
                    f"User: {h['user']}\nAssistant: {h['assistant']}"
                    for h in session.history[-2:]
                ]  # Last 2 exchanges
            )
            if session.history
            else "No previous conversation"
        )

        scheme_context = (
            "\n\n".join(
                [
                    self.create_scheme_text(s) for s in session.last_schemes[:2]
                ]  # Top 2 schemes
            )
            if session.last_schemes
            else "No relevant schemes found"
        )

        return f"""**Strict Response Guidelines**
    1. Respond ONLY using information from these schemes:
    {scheme_context}

    2. If asked about anything outside these schemes, respond:
    "I specialize in government schemes. Could you ask about available programs like [mention relevant scheme]?"

    3. Never mention political figures, parties, or non-scheme entities

    4. For unrelated queries:
    "This question is outside my expertise. I can help with government schemes like [scheme1], [scheme2] etc."

    **Conversation History**
    {history_context}

    **Current Query**
    {query}

    **Required Response Format**
    - Use markdown bullet points
    - Always mention scheme names
    - Include eligibility if relevant
    - Never make up information
    - Redirect to schemes if unsure"""


if __name__ == "__main__":
    chatbot = GovtSchemeChatbot()
    print("Chatbot initialized. Type 'exit' to quit.")

    while True:
        try:
            query = input("\nYou: ")
            if query.lower() in ["exit", "quit"]:
                break

            response = chatbot.generate_response(query)
            print("\nAssistant:", response)

        except KeyboardInterrupt:
            print("\nSession ended.")
            break
        except Exception as e:
            print(f"Critical error: {str(e)}")
            chatbot = GovtSchemeChatbot()  # Reset chatbot

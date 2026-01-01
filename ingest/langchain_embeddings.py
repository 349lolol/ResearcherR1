from langchain_google_genai import GoogleGenerativeAIEmbeddings

class GeminiEmbeddings:
    def __init__(self, model: str = "models/embedding-001"):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model = model,
            task_type = "RETRIVAL_DOCUMENT"
        )
    def get_langchain_embeddings(self):
        return self.embeddings
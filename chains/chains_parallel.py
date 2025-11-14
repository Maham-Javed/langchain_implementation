from dotenv import load_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_groq import ChatGroq
from langchain_core.output_parsers.string import StrOutputParser


load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Use the updated model
model = ChatGroq(
    api_key=api_key,
    model="llama-3.1-8b-instant", 
    temperature=0.7 # temperature controls the randomness or creativity of the model’s responses
)

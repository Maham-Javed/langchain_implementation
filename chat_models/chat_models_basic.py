
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Use the updated model
model = ChatGroq(
    api_key=api_key,
    model="llama-3.1-8b-instant", 
    temperature=0.7 # temperature controls the randomness or creativity of the model’s responses
)
# invoke the model by using invoke function
result = model.invoke("What is square root of 46.4?")

# Show the full result
print("Full Result:")
print(result)

# show the whole content of the result
print("Content:")
print(result.content)
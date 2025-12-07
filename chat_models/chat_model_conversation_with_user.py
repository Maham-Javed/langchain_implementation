
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Use the updated model
model = ChatGroq(
    api_key=api_key,
    model="llama-3.1-8b-instant", 
    temperature=0.7 # temperature controls the randomness or creativity of the model’s responses
)

# Initialize the chat history
chat_history = []

# write system message for giving the context to the LLM and store it in a chat history as a first message
system_Message = SystemMessage(content="You are an helpfull AI assitence.")
chat_history.append(system_Message)

while True:
    # taking input form user
    query = input("User: ")
    if query.lower == "exit":
        break

    # adding the user question to the chat history
    humanMessage = HumanMessage(content=query)
    chat_history.append(humanMessage)

    # providing the entire chat history to the model for better context
    result = model.invoke(chat_history)

    # Extracts the textual content from the returned model object
    response = result.content

    # Appends the assistant’s reply to chat_history so it will be included in future turns.
    chat_history.append(AIMessage(content=response))

    # Prints the assistant’s reply to the console.
    print(f"\nAI: {response}")

print("\n\n----------Chat History----------\n\n")
# printing the entire chat history between the LLM and user
print(f"{chat_history}")
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableSequence

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Use the updated model
model = ChatGroq(
    api_key=api_key,
    model="llama-3.1-8b-instant", 
    temperature=0.7 # temperature controls the randomness or creativity of the model’s responses
)
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a comedian who tells jokes about {topic}."), # System message defines the assistant’s role or behavior
    ("human", "Tell me {joke_count} jokes."), # Human message gives the user’s instruction
])

# # Create individual runnables (steps in the chain)
format_prompt = RunnableLambda(lambda x : prompt_template.format_prompt(**x))
invoke_model = RunnableLambda(lambda x : model.invoke(x.to_messages()))
parse_output = RunnableLambda(lambda x : x.content)

# Create the RunnableSequence (equivalent to the LCEL chain)
# middle = [] -> this will add all the chains in between (if any)
chain = RunnableSequence(first=format_prompt, middle=[invoke_model], last=parse_output)

# Run the chain
response = chain.invoke({"topic": "lawyers", "joke_count": 3})

# print output
print(response)
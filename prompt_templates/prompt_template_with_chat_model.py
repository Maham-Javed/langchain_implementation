
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Use the updated model
model = ChatGroq(
    api_key=api_key,
    model="llama-3.1-8b-instant", 
    temperature=0.7 # temperature controls the randomness or creativity of the model’s responses
)
# PART 1: Create a ChatPromptTemplate using a template string

# Define a text template with a placeholder `{topic}`, this placeholder fiiled later with the actual value when invoked.
template = "Tell me a joke about {topic}."

# Create a ChatPromptTemplate object from the template string 
prompt_template = ChatPromptTemplate.from_template(template)

print("---- Prompt from template ----")
# Invoke the prompt_template by passing a dictionary 
prompt = prompt_template.invoke({"topic": "cat"})
# imvole the model for answer the prompt
result = model.invoke(prompt)
print(result.content)
# PART 2: Prompt with Multiple Placeholders

template = "Tell me {number} jokes about {topic}."
prompt_template = ChatPromptTemplate.from_template(template)

print("---- Prompt from template ----")
prompt = prompt_template.invoke({"number": 3, "topic": "cat"})
# imvole the model for answer the prompt
result = model.invoke(prompt)
print(result.content)
# PART 3: Prompt with System and Human Messages (Using Tuples)

# Define a list of message tuples.
messages = [
    ("system", "You are a comedian who tells jokes about {topic}."), # System message defines the assistant’s role or behavior
    ("human", "Tell me {joke_count} jokes."), # Human message gives the user’s instruction
]

# Create a ChatPromptTemplate from the list of messages.
# LangChain will recognize placeholders like {topic} and {joke_count}
# and let you fill them later using a dictionary.
prompt_template = ChatPromptTemplate.from_messages(messages)

print("---- Prompt from template ----")
prompt = prompt_template.invoke({"topic": "lawyers", "joke_count": 3})
# imvole the model for answer the prompt
result = model.invoke(prompt)
print(result.content)
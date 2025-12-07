from dotenv import load_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch
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

# Define prompt templates for different feedback types:
# Positive Feedback:
positive_feedback_template = ChatPromptTemplate.from_messages(
    [
        ('system', "You are a helpful assistant."),
        ("human", "Generate a thank you note for this positive feedback: {feedback}.")
    ]
)

# Negative Feedback:
negative_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "Generate a response addressing this negative feedback: {feedback}."),
    ]
)

# Neutral Feedback:
neutral_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("Generate a request for more details for this neutral feedback: {feedback}."),
    ]
)

# Escalate Feedback:
escalate_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("Generate a request for more details for this neutral feedback: {feedback}."),
    ]
)

# Define the feedback classification template
classification_template_feedback = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistent."),
        ("human", "Classify the sentiment of this feedback as positive, negative, neutral, or escalate: {feedback}.")
    ]
)

# Define the runnable branches for handling feedbacks - positive, negative, neutral and escalate
branches = RunnableBranch(
    (
        lambda x: 'positive' in x,
        positive_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x: 'negative' in x,
        negative_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x: 'neutral' in x,
        neutral_feedback_template | model | StrOutputParser()
    ),
    escalate_feedback_template | model | StrOutputParser()
)

# Create the classification chain
classification_chain = classification_template_feedback | model | StrOutputParser()

# Combine classification and response generation into one chain
chain = classification_chain | branches

review = "The product is terrible. It broke after just one use and the quality is very poor."
result = chain.invoke({"feedback": review})

# Output the result
print(result)

# Run the chain with an example review
# Positive review - "The product is excellent. I really enjoyed using it and found it very helpful."
# Negative review - "The product is terrible. It broke after just one use and the quality is very poor."
# Neutral review - "The product is okay. It works as expected but nothing exceptional."
# Escalate - "I'm not sure about the product yet. Can you tell me more about its features and benefits?"
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Use the updated model
model = ChatGroq(
    api_key=api_key,
    model="llama-3.1-8b-instant", 
    temperature=0.7 # temperature controls the randomness or creativity of the model’s responses
)
# define prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
    ("system", "You are a comedian who tells jokes about {topic}."), # System message defines the assistant’s role or behavior
    ("human", "Tell me {joke_count} jokes."), # Human message gives the user’s instruction
    ]
)

# defining a series of Runnable functions to transform the data one after the other

uppercase_jokes = RunnableLambda(
    lambda x: x.upper()
)  # Uppercase the generated content from the LLM
count_words = RunnableLambda(
    lambda x: f"\nNumber of Words are : {len(x.split())}\n\n\n{x}"
)  # count the words and print the result
# This will define the chain of execution of those Runnables
chain = (prompt_template | model | StrOutputParser() | uppercase_jokes | count_words )

# invoking the chain and also providing the variables to the prompt_template we defined above
result = chain.invoke({"topic": "lawyer", "joke_count": 3})
print(result)
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint,HuggingFacePipeline
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Ensure the Hugging Face API token is set
if "HUGGINGFACEHUB_API_TOKEN" not in os.environ:
    raise ValueError("Please set the HUGGINGFACEHUB_API_TOKEN in your environment.")

# Initialize the HuggingFaceEndpoint
llm = HuggingFacePipeline.from_model_id(
    model_id="google/gemma-2-27b-it",  # Ensure this is the correct model name
    task="text-generation"
)

# Initialize the chat model
model = ChatHuggingFace(llm=llm)

# Invoke the model
try:
    result = model.invoke("what is the capital of india")
    print(result.content)
except Exception as e:
    print(f"An error occurred: {e}")
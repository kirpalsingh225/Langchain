from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence


load_dotenv()


prompt1 = PromptTemplate(
    template = "Write a joke about {topic}",
    input_variables = ["topic"]
)

prompt2 = PromptTemplate(
    template="Explain the following joke - {text}",
    input_variables = ["text"]
)

parser = StrOutputParser()

model = ChatGoogleGenerativeAI(model = 'gemini-2.0-flash', api_key="AIzaSyCuE7hhVc-h-fuim6iP5eVYHM_32FiZf98")


chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)

result = chain.invoke({"topic":"cricket"})
print(result)
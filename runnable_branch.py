from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import (RunnableSequence, RunnableParallel,
 RunnablePassthrough, RunnableLambda, RunnableBranch)


load_dotenv()



prompt1 = PromptTemplate(
    template = "Write a detailed report on {topic}",
    input_variables = ["topic"]
)

prompt2 = PromptTemplate(
    template="Summarize the following text \n {text}",
    input_variables = ["text"]
)

parser = StrOutputParser()

model = ChatGoogleGenerativeAI(model = 'gemini-2.0-flash', api_key="AIzaSyCuE7hhVc-h-fuim6iP5eVYHM_32FiZf98")


report_generator = RunnableSequence(prompt1, model, parser)

branch_chain = RunnableBranch(
    #(condition, runnable)
    (lambda x : len(x.split()) > 500, RunnableSequence(prompt2, model, parser) ),   #no of tuples equal to the no of tuples as input
    #default type of else
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_generator, branch_chain)

result = final_chain.invoke({"topic":"taylor swift"})
print(result)
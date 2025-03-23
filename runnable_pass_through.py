from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough


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

model = ChatGoogleGenerativeAI(model = 'gemini-2.0-flash')


joke_generator = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel(
    {
        "joke" : RunnablePassthrough(),
        "explanation" : RunnableSequence(prompt2, model, parser)
    }
)

final_chain = RunnableSequence(joke_generator, parallel_chain)

result = final_chain.invoke({"topic":"taylor swift"})
print(result)

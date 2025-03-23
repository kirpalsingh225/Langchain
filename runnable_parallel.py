from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel


load_dotenv()


prompt1 = PromptTemplate(
    template = "Generate a tweet about {topic}",
    input_variables = ["topic"]
)

prompt2 = PromptTemplate(
    template="Generate a linkedin post about - {topic}",
    input_variables = ["topic"]
)

parser = StrOutputParser()

model = ChatGoogleGenerativeAI(model = 'gemini-2.0-flash', api_key="AIzaSyCuE7hhVc-h-fuim6iP5eVYHM_32FiZf98")


chain = RunnableParallel(
    {
        "tweet" : RunnableSequence(prompt1, model, parser),
        "x" : RunnableSequence(prompt2, model, parser)
    }
)

#output will be of the type string 
# tweet -> prompt - mode - parser | answer as dict {"tweer":, "X"}
# X -> prompt - model - parser    |

result = chain.invoke({"topic":"taylor swift"}) #this input will be given to both runnable sequence
print(result)
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal


load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-2.0-flash')

parser = StrOutputParser()


class FeedBack(BaseModel):

    sentiment : Literal["positive", "negative"] = Field(desc = "Give the sentiment of the feedback")


parser2 = PydanticOutputParser(pydantic_object=FeedBack)


prompt1 = PromptTemplate(
    template = "Classify the sentiment of the following feedback into positive or negative \n {feedback} \n {format_instruction}",
    input_variables = ["feedback"],
    partial_variables = {"format_instruction":parser2.get_format_instructions()}
)


classifier_chain = prompt1 | model | parser2


prompt2 = PromptTemplate(
    template = "Write appropriate response for the positive feedback \n {feedback}",
    input_variables = ["feedback"]
)

prompt3 = PromptTemplate(
    template = "Write appropriate response for the negative feedback \n {feedback}",
    input_variables = ["feedback"]
)
branch_chain = RunnableBranch(
    #(condition, statment if condition is true)
    (lambda x:x.sentiment=="positive", prompt2 | model | parser),
    (lambda x:x.sentiment=="negative", prompt3| model| parser),
    RunnableLambda(lambda x : "could not find sentiment")
)


chain = classifier_chain | branch_chain

print(chain.invoke({"feedback" : "this is a terrible phone it does not work well"}))

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from pydantic import BaseModel, Field

load_dotenv()


class Person(BaseModel):

    name : str = Field(description="name of the person")
    age : int = Field(gt=18, description="age of the person")
    city : str = Field(description="city where the person belongs from")

parser = PydanticOutputParser(pydantic_object = Person)

prompt = PromptTemplate(
    template = "Generate name, age and city of a fictional {place} person \n {format_instruction}",
    input_variables = ["place"],
    partial_variables = {'format_instruction' : parser.get_format_instructions()}
)

fromatted_prompt = prompt.invoke({"place: indian"})

model = ChatGoogleGenerativeAI(model = 'gemini-2.0-flash')

result = model.invoke(fromatted_prompt)

final_result = parser.parse(result.content)

print(final_result)


#we cannot do data validation using this parser
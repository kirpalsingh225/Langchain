from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()


schema = [
    ResponseSchema(name="fact_1", description="fact 1 about the topic"),
    ResponseSchema(name="fact_2", description="fact 2 about the topic"),
    ResponseSchema(name="fact_3", description="fact 3 about the topic")
]

parser = StructuredOutputParser.from_response_schemas(schema)

prompt = PromptTemplate(
    template = "Generate 5 interesting facts about {topic} \n {format_instruction}",
    input_variables = ["topic"],
    partial_variables = {'format_instruction' : parser.get_format_instructions()}
)

fromatted_prompt = prompt.invoke({"input: taylor swift"})

model = ChatGoogleGenerativeAI(model = 'gemini-2.0-flash')

result = model.invoke(fromatted_prompt)

final_result = parser.parse(result.content)

print(final_result)


#we cannot do data validation using this parser
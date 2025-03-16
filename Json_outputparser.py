from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


load_dotenv()

parser = JsonOutputParser()

prompt = PromptTemplate(
    template = "Generate 5 interesting facts about {topic} \n {format_instruction}",
    input_variables = ["topic"],
    partial_variables = {'format_instruction' : parser.get_format_instructions()}
)

fromatted_prompt = prompt.format(topic="taylor swift")

model = ChatGoogleGenerativeAI(model = 'gemini-2.0-flash')

result = model.invoke(fromatted_prompt)

final_answer = parser.parse(result.content)
print(final_answer["taylor_swift_facts"])
print(type(final_answer))

import getpass
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()

def get_llm_translator():
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")
    return ChatOpenAI(model_name="gpt-4.1-nano", temperature=0)


prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that translates German to English only if the language is german. Else return the original text."),
    ("user", "{text}")
])


def translate_text(text: str) -> str:
    try:
        llm_translator = get_llm_translator()
        prompt = prompt_template.invoke({"text": text})
        response = llm_translator.invoke(prompt)
        return response.content
    except Exception as e:
        print("Exception in translate_text:", e)
        return "Translation failed due to error: " + str(e)

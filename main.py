from dotenv import load_dotenv
from importlib.metadata import version
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

core_version = version("langchain-core")
lg_version = version("langgraph")
print(f"langchain-core version: {core_version}")
print(f"langgraph version: {lg_version}")


def main():
    
    # Test gemini
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")
    response_gemini = llm.invoke("Say 'setup complete!' in one word")
    print(f"Response from ChatGemini: {response_gemini}")

    print("Setup complete!")


if __name__ == "__main__":
    main()

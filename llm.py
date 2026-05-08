from langchain_openai import ChatOpenAI
from settings import Settings

settings = Settings()

llm = ChatOpenAI(
    model = settings.default_model,
    temperature = 0,
    base_url = settings.proxy_base_url
    )
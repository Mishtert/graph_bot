import os

from dotenv import load_dotenv
from langchain_openai.chat_models.azure import AzureChatOpenAI
from langchain_openai.embeddings.azure import AzureOpenAIEmbeddings


_ = load_dotenv('C:/Users/319533/OneDrive - Cognizant/Documents/01.App/01.2023/gen_ai_app/env_config/fn_api_keys.env')

# Load API Keys & Model
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
model = AzureChatOpenAI(temperature=0.3,
                        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                        azure_deployment=os.getenv("AZURE_DEPLOYMENT"),
                        verbose=True)

embeddings = AzureOpenAIEmbeddings(chunk_size=1,
                                   openai_api_version=os.getenv("EMBEDDINGS_API_VERSION"),
                                   azure_endpoint=os.getenv("EMBEDDINGS_ENDPOINT"),
                                   openai_api_key=os.getenv("EMBEDDINGS_API_KEY"),
                                   model=os.getenv("EMBEDDINGS_MODEL"),
                                   deployment=os.getenv("EMBEDDINGS_DEPLOYMENT")
                                   )
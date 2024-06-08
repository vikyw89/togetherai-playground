import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
import instructor

load_dotenv()


togetherai_client = AsyncOpenAI(
    api_key=os.environ.get("TOGETHER_API_KEY"),
    base_url=os.environ.get("TOGETHER_API_BASE_URL"),
)
structured_togetherai_client = instructor.from_openai(
    client=togetherai_client, mode=instructor.Mode.TOOLS
)
openai_client = AsyncOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)
structured_openai_client = instructor.from_openai(client=openai_client)

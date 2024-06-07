import os
from dotenv import load_dotenv
from together import AsyncTogether
load_dotenv()

client = AsyncTogether(api_key=os.environ.get("TOGETHER_API_KEY"))


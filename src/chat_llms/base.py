from abc import abstractmethod
from openai import AsyncStream
from instructor.client import T
from src.prompts.system_messages import ASSISTANT
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionMessage, ChatCompletionChunk

    
class BaseChatLLM:
    def __init__(
        self,
        model: str,
        verbose: bool = False,
        messages: list[ChatCompletionMessageParam] = [],
    ) -> None:
        self.model = model
        self.verbose = verbose
        self.messages = messages
        self.tools = []

    @abstractmethod
    async def arun(self) -> ChatCompletionMessage:
        pass

    def add_message(self, message: ChatCompletionMessageParam) -> None:
        self.messages.append(message)

    @abstractmethod
    async def astream(self) -> AsyncStream[ChatCompletionChunk]:
        pass

    @abstractmethod
    async def astructured_extraction(self, output_class: type[T]) -> T:
        pass
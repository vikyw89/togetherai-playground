from openai import AsyncStream
from src.prompts.system_messages import ASSISTANT
from src.chat_llms.base import BaseChatLLM
from src.configs import openai_client, structured_openai_client
from src.chat_llms.base import (
    ChatCompletionMessage,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    T,
)


class ChatOpenaiLLM(BaseChatLLM):
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        verbose: bool = False,
        messages: list[ChatCompletionMessageParam] = [],
    ) -> None:
        return super().__init__(model=model, verbose=verbose, messages=messages)

    async def arun(self) -> ChatCompletionMessage:
        response = await openai_client.chat.completions.create(
            messages=self.messages,
            model=self.model,
        )
        return response.choices[0].message

    async def astream(self) -> AsyncStream[ChatCompletionChunk]:
        stream = await openai_client.chat.completions.create(
            messages=self.messages, model=self.model, stream=True
        )
        return stream

    async def astructured_extraction(self, output_class: type[T]) -> T:
        response = await structured_openai_client.chat.completions.create(
            messages=self.messages,
            model=self.model,
            max_retries=3,
            response_model=output_class,
        )
        return response

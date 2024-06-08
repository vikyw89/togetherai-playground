from typing import AsyncGenerator
from src.llms.base import BaseLLM, T
from src.prompts.system_messages import ASSISTANT
from src.configs import openai_client, structured_openai_client


class OpenaiLLM(BaseLLM):

    def __init__(
        self,
        prompt: str = ASSISTANT,
        model: str = "gpt-3.5-turbo",
        verbose: bool = False,
    ) -> None:
        super().__init__(model=model, prompt=prompt, verbose=verbose)

    async def arun(self, text: str) -> str:
        response = await openai_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": text}],
        )

        return response.choices[0].message.content or ""

    async def astream(self, text: str) -> AsyncGenerator[str, None]:
        stream = await openai_client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": text}], stream=True
        )

        async for chunk in stream:
            delta_content = chunk.choices[0].delta.content or ""
            yield delta_content

    async def astructured_extraction(self, text: str, output_class: type[T]) -> T:
        response = await structured_openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "Let's think step by step. Given a text, extract structured data from it.",
                },
                {"role": "user", "content": text},
            ],
            max_retries=3,
            response_model=output_class,
        )
        return response

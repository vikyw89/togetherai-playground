from src.llms.together_ai import TogetherAILLM

async def main() -> None:
    llm = TogetherAILLM()

    stream = llm.astream("What is the meaning of life?")

    async for chunk in stream:
        print(chunk,end="")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
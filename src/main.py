import time
from src.llms.together_ai import TogetherAILLM
from together.types import ChatCompletionChunk

async def main() -> None:
    llm = TogetherAILLM()
    start_time = time.time()
    stream = await llm.astream("What is the meaning of life?")
    
    # create variables to collect the stream of chunks
    collected_chunks = []
    collected_messages = []
    # iterate through the stream of events
    async for chunk in stream:
        chunk_time = time.time() - start_time  # calculate the time delay of the chunk
        collected_chunks.append(chunk)  # save the event response
        if chunk.choices is None:
            continue
        if chunk.choices[0].delta is None:
            continue
        chunk_message = chunk.choices[0].delta.content  # extract the message
        collected_messages.append(chunk_message)  # save the message
        print(f"Message received {chunk_time:.2f} seconds after request: {chunk_message}")  # print the delay and text

    # print the time delay and text received
    print(f"Full response received {chunk_time:.2f} seconds after request")
    # clean None in collected_messages
    collected_messages = [m for m in collected_messages if m is not None]
    full_reply_content = ''.join(collected_messages)
    print(f"Full conversation received: {full_reply_content}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
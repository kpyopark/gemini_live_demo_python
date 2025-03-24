import asyncio
from google import genai
from dotenv import load_dotenv
import numpy as np
import os
import io
import wave
import sounddevice as sd
import time  # time 모듈 유지 (필요에 따라 제거 가능)
import asyncio # asyncio Queue 사용, 이미 import 되어 있음
import traceback # 에러 추적을 위해 추가

from google.genai.types import (
    Part,
    FunctionDeclaration,
    GoogleSearch,
    LiveConnectConfig,
    PrebuiltVoiceConfig,
    SpeechConfig,
    Content,
    Tool,
    ToolCodeExecution,
    VoiceConfig,
    FunctionResponse,
    LiveClientToolResponse
)

load_dotenv()

GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")


client = genai.Client(api_key=GEMINI_API_KEY, http_options={'api_version': 'v1alpha'})
model_id = "gemini-2.0-flash-exp"

def get_current_weather_impl(location) -> dict[str, str] :
    """Get the current weather in a given location"""
    print(f"Getting weather for {location}")
    return {
        "temperature": "72",
        "unit": "fahrenheit",
        "description": "sunny"
    }

get_current_weather = FunctionDeclaration(
    name="get_current_weather",
    description="Get the current weather in a given location",
    parameters={
        "type": "OBJECT",
        "properties": {"location": {"type": "STRING", "description": "Location"}},
    }
)

def log_conversation_in_history_impl(conversation) -> dict[str, str] :
    """Get the conversation history log"""
    print(f"Getting conversation history log for {conversation}")
    return {
        "log": "success"
    }

log_conversation_in_history = FunctionDeclaration(
    name="log_conversation_in_history",
    description="Before to talk, please log the planned conversation you will talk. ",
    parameters={
        "type": "OBJECT",
        "properties": {"conversation": {"type": "STRING", "description": "statements for AI assistant to talk"}},
    }
)

system_instruction = """You are a helpful Korean AI assistant with multimodal capabilities.

You have the following tools available to you:
- get_current_weather: Get current weather information for a city

Rules:
- You are in Korea. You should say in Korean when you read numbers.
- Whenever you're asked about the weather you MUST use the get_current_weather tool.
- Please speak in Korean.
- You should read numbers in Korean. For example, 2.0 should be read as "이점영.". Gemini should be read as "제미니.". "22도" should be read as "이십이도."
- You should log the conversation history before you talk. You can use the log_conversation_in_history tool to do this.
- If you have already log your conversation into history, do not repeat the same conversation into speaking.
- Do not repeat the same conversation into speaking.
"""

generation_config = {
    "temperature": 0.35,
}

config = LiveConnectConfig(
    response_modalities=["AUDIO"],
    tools=[Tool(function_declarations=[get_current_weather, log_conversation_in_history], code_execution={})],
    system_instruction=Content(role="model", parts=[Part(text=system_instruction)]),
    speech_config=SpeechConfig(
        voice_config=VoiceConfig(
            prebuilt_voice_config=PrebuiltVoiceConfig(
                voice_name="Aoede",
            )
        )
    ),
    generation_config=generation_config
)

MODEL = "models/gemini-2.0-flash-exp"
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 2048
DTYPE = np.int16  # Define the data type for audio
CHANNELS = 1 # Mono audio


async def main():
    audio_queue = asyncio.Queue()
    model_speaking = False
    session = None

    try:
        async with (
            client.aio.live.connect(model=MODEL, config=config) as session, # config를 소문자로 수정
            asyncio.TaskGroup() as tg,
        ):

            async def listen_and_send():
                print('start listen_and_send')
                nonlocal model_speaking
                with sd.InputStream(samplerate=SEND_SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE) as input_stream:
                    while True:
                        if not model_speaking:
                            try:
                                data, overflowed = await asyncio.to_thread(input_stream.read, CHUNK_SIZE)
                                #data, overflowed = input_stream.read(CHUNK_SIZE) # 직접 read 호출, overflowed 변수 추가
                                if overflowed:
                                    print("Audio input overflowed!")
                                # data는 numpy array이므로 bytes로 변환해야 함
                                data_bytes = data.tobytes()
                                #print('audio input received. length of byte : ', len(data_bytes))
                                await session.send(input={"data": data_bytes, "mime_type": "audio/pcm"},end_of_turn=True)
                            except OSError as e:
                                print(f"Audio input error: {e}")
                                await asyncio.sleep(0.1)
                        else:
                            await asyncio.sleep(0.1)

            async def receive_and_play():
                print('start receive_and_play')
                nonlocal model_speaking
                with sd.OutputStream(samplerate=RECEIVE_SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE) as output_stream:
                    print('start receiveing response.')
                    while True:
                        async for response in session.receive():
                            print('response received')
                            server_content = response.server_content
                            if server_content and server_content.model_turn:
                                model_speaking = True
                                for part in server_content.model_turn.parts:
                                    if part.inline_data:
                                        # inline_data.data는 bytes이므로 numpy array로 변환 후 write
                                        audio_data = np.frombuffer(part.inline_data.data, dtype=DTYPE)
                                        await asyncio.to_thread(output_stream.write, audio_data)
                                        #output_stream.write(audio_data) # await 불필요
                            if response.tool_call:
                                print('tool call received.')
                                # id='function-call-3494351139624454581' args={'location': '서울'} name='get_current_weather'
                                function_responses = []
                                for function_call in response.tool_call.function_calls:
                                    print(function_call)
                                    if function_call.name == "get_current_weather":
                                        if 'location' in function_call.args:
                                            location = function_call.args['location']
                                        else:
                                            location = "Seoul"
                                        function_response = FunctionResponse(
                                            name=function_call.name,
                                            id=function_call.id,
                                            response={
                                                "result" : {
                                                    "object_value" : get_current_weather_impl(location=location)
                                                }
                                            }
                                        )
                                        function_responses.append(function_response)
                                    elif function_call.name == "log_conversation_in_history":
                                        if 'conversation' in function_call.args:
                                            conversation = function_call.args['conversation']
                                        else:
                                            conversation = "None"
                                        function_response = FunctionResponse(
                                            name=function_call.name,
                                            id=function_call.id,
                                            response={
                                                "result" : {
                                                    "object_value" : log_conversation_in_history_impl(conversation=conversation)
                                                }
                                            }
                                        )
                                        function_responses.append(function_response)


                                live_function_response = LiveClientToolResponse(function_responses=[function_response])
                                print(live_function_response)
                                await session.send(input=live_function_response, end_of_turn=True)

                            if server_content and server_content.turn_complete:
                                print("Turn complete")
                                model_speaking = False
                                
            print('create listen and send task')
            tg.create_task(listen_and_send())
            print('create receive and play task')
            tg.create_task(receive_and_play())

    except Exception as e:
        traceback.print_exception(None, e, e.__traceback__)


if __name__ == "__main__":
    asyncio.run(main(), debug=True)
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

system_instruction = """You are a helpful AI assistant with multimodal capabilities.

You have the following tools available to you:
- get_current_weather: Get current weather information for a city

Rules:
- Whenever you're asked about the weather you MUST use the get_current_weather tool.
"""

config = LiveConnectConfig(
    response_modalities=["AUDIO"],
    tools=[Tool(function_declarations=[get_current_weather], code_execution={})],
    system_instruction=Content(role="model", parts=[Part(text=system_instruction)]),
    speech_config=SpeechConfig(
        voice_config=VoiceConfig(
            prebuilt_voice_config=PrebuiltVoiceConfig(
                voice_name="Aoede",
            )
        )
    ),
)

SAMPLE_RATE=24000
DTYPE = np.int16  # Define the data type for audio
CHANNELS = 1 # Mono audio


async def main():
    is_playing = False
    playback_task = None # playback_task 로 변경 (asyncio Task)
    response_task = None
    input_task = None # input_task 로 변경 (asyncio Task)
    audio_queue = asyncio.Queue() # asyncio Queue 사용
    input_queue = asyncio.Queue() # asyncio Queue 사용
    
    async def playback_worker(): # async function 으로 변경
        """오디오 큐에서 청크를 가져와 스트림 방식으로 재생하는 비동기 워커 함수"""
        stream = None  # 스트림 객체 초기화

        try:
            stream = await asyncio.to_thread( # asyncio.to_thread 사용
                sd.OutputStream,
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=DTYPE
            )
            stream.start() # 스트림 시작
            print("오디오 스트림 시작 (asyncio.to_thread)")

            while True:
                try:
                    audio_chunk_bytes = await audio_queue.get() # await audio_queue.get() 사용 (asyncio Queue)
                    if audio_chunk_bytes is None:  # 종료 신호
                        print("종료 신호 받음: 오디오 스트림 종료 (asyncio.to_thread)")
                        break

                    audio_chunk_np = np.frombuffer(audio_chunk_bytes, dtype=DTYPE)
                    print('audio_chunk recieved.... (asyncio.to_thread)')
                    if audio_chunk_np.size > 0:
                        await asyncio.to_thread(stream.write, audio_chunk_np) # asyncio.to_thread 사용
                        is_playing = True
                    else:
                        print("수신된 오디오 데이터 없음 (빈 청크) (asyncio.to_thread)")
                    # audio_queue.task_done() # asyncio.Queue 에서는 task_done() 불필요

                except sd.PortAudioError as e:
                    print(f"PortAudio 에러: {e} (asyncio.to_thread)")
                    is_playing = False
                    break
                except Exception as e:
                    print(f"오디오 재생 중 에러 발생: {e} (asyncio.to_thread)")
                    is_playing = False
                    traceback.print_exc() # 상세 에러 출력
                    break

        except sd.PortAudioError as stream_error:
            print(f"스트림 생성 에러: {stream_error} (asyncio.to_thread)") # 스트림 생성 에러 처리
        except Exception as e_stream_init:
            print(f"스트림 초기화 중 에러 발생: {e_stream_init} (asyncio.to_thread)")
            traceback.print_exc() # 상세 에러 출력
        finally:
            if stream and stream.active:
                stream.stop()
                stream.close() # 스트림 닫기 (자원 해제)
                print("오디오 스트림 중지 및 닫힘 (asyncio.to_thread)")
            is_playing = False
            print("오디오 재생 task 종료 (asyncio.to_thread)")

    async def process_response(session):
        """ Gemini response를 비동기적으로 수신하고 오디오 큐에 넣는 task (asyncio Task) """
        print('start process_response')
        async for message in session.receive(): # while True 루프 제거
            print('message received!!!!!!!!!!!!!!!')
            #print(message)

            if message.tool_call:
                for function_call in message.tool_call.function_calls:
                    print(function_call)
                    # tool_response : 
                    #     function_responses : Array(1)
                    #         0 : 
                    #             id : "function-call-3937553967149120411"
                    #             name : "get_stock_price"
                    #             response : 
                    #                 result : 
                    #                     object_value : 
                    #                         error : "Error fetching stock price for AAPL: Finnhub API failed with status: 401"
                    function_response = FunctionResponse(
                        name=function_call.name,
                        id=function_call.id,
                        response={
                            "result" : {
                                "object_value" : get_current_weather_impl("Seoul")
                            }
                        }
                    )
                    live_function_response = LiveClientToolResponse(function_responses=[function_response])
                    print(live_function_response)
                    await input_queue.put(live_function_response) # await input_queue.put 사용 (asyncio Queue)
                    # await session.send(live_function_response, end_of_turn=True) # input_queue로 전달 후 main()에서 session.send 처리

            if message.server_content.model_turn:
                for part in message.server_content.model_turn.parts:
                    if part.inline_data:
                        audio_chunk_bytes = part.inline_data.data # bytes 데이터
                        print('audio chunk recieved....')
                        #part.inline_data.data = None
                        #print(part)
                        await audio_queue.put(audio_chunk_bytes) # await audio_queue.put 사용 (asyncio Queue)
                    if part.excutable_code:
                        excutable_code = part.excutable_code
                        print(excutable_code.code)
                        print(excutable_code.language)

            if message.server_content.turn_complete:
                print('turn complete received.')
                print(message.server_content)
                # await audio_queue.put(None) # 종료 신호 불필요
                # response_task = None # response_task 초기화 불필요, 함수 유지


        print('stop process_response') # 이 부분은 session.receive() 루프가 끝나면 도달 (예: 연결 종료)


    async def input_worker(): # async function 으로 변경
        """ 사용자 입력을 비동기적으로 받는 워커 함수 (asyncio Task) """
        while True:
            message = await asyncio.to_thread(input, "User> ") # asyncio.to_thread 로 blocking input() 호출
            await input_queue.put(message) # await input_queue.put 사용 (asyncio Queue)
            if message.lower() == "exit":
                break
        print("입력 task 종료 (asyncio.to_thread)")

    playback_task = asyncio.create_task(playback_worker()) # asyncio.create_task 로 task 시작
    input_task = asyncio.create_task(input_worker()) # asyncio.create_task 로 task 시작
    response_task = None

    async with client.aio.live.connect(model=model_id, config=config) as session:
        print("Gemini Live API 연결됨. 'exit'를 입력하여 종료.")
        response_task = asyncio.create_task(process_response(session)) # task 시작 및 변수에 저장

        while True:
            try:
                message = await input_queue.get() # await input_queue.get() 사용 (asyncio Queue)
                # input_queue.task_done() # asyncio.Queue 에서는 task_done() 불필요

                if isinstance(message, str): # message가 문자열인지 확인
                    if message.lower() == "exit":
                        break
                    await session.send(input=message, end_of_turn=True)
                elif isinstance(message, LiveClientToolResponse): # LiveClientToolResponse 객체인 경우
                    print(f"Sending function response: {message}")
                    await session.send(live_client_tool_response=message, end_of_turn=True) # live_client_tool_response 파라미터 사용
                else:
                    # 문자열이 아닌 경우의 처리 (예: 로깅, 무시, 다른 방식의 처리)
                    print(f"Received unexpected message type: {type(message)}, message: {message}")
                    # await session.send(input=message) # 필요에 따라 수정. 현재는 unexpected type 이므로 send 하지 않음.
            except Exception as e: # queue.Empty 예외 처리 제거. asyncio.Queue.get()은 Queue가 비어있을 때 await 상태가 됨
                print(f"Error while getting message from queue: {e}")
                traceback.print_exc() # 상세 에러 출력
                continue # or break, depending on error handling strategy

    print("exit command received. waiting for tasks to complete...") # 종료 대기 메시지

    
    if playback_task is not None and not playback_task.cancelled():
        await audio_queue.put(None) # 마지막 종료 신호, await audio_queue.put 사용 (asyncio Queue)
        await playback_task # playback_task 완료까지 기다림
        print("최종 재생 task 종료 대기 완료 (asyncio.to_thread)")


    print("프로그램 종료")


if __name__ == "__main__":
    asyncio.run(main(), debug=True)
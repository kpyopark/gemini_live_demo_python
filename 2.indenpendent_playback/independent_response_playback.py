import asyncio
from google import genai
from dotenv import load_dotenv
import numpy as np
import os
import io
import wave
import sounddevice as sd
import threading
import time  # time 모듈 추가
import queue # Use standard queue for thread-safe communication

load_dotenv()

GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")


client = genai.Client(api_key=GEMINI_API_KEY, http_options={'api_version': 'v1alpha'})
model_id = "gemini-2.0-flash-exp"
config = {"responseModalities": ["TEXT"]}

SAMPLE_RATE=24000
DTYPE = np.int16  # Define the data type for audio
CHANNELS = 1 # Mono audio

audio_queue = asyncio.Queue() # 오디오 청크(bytes)를 저장할 큐
input_queue = queue.Queue() # 입력 큐 (쓰레드-asyncio 통신용)
is_playing = False
playback_thread = None
response_task = None # 응답 처리 task를 저장할 변수
input_thread = None # 입력 쓰레드


def playback_worker():
    """오디오 큐에서 청크를 가져와 스트림 방식으로 재생하는 워커 쓰레드 함수 (non-async)"""
    global is_playing
    stream = None  # 스트림 객체 초기화

    try:
        stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE)
        stream.start() # 스트림 시작
        print("오디오 스트림 시작")

        while True:
            try:
                audio_chunk_bytes = audio_queue.get_nowait()
                if audio_chunk_bytes is None:  # 종료 신호
                    print("종료 신호 받음: 오디오 스트림 종료")
                    break

                audio_chunk_np = np.frombuffer(audio_chunk_bytes, dtype=DTYPE)

                if audio_chunk_np.size > 0:
                    stream.write(audio_chunk_np) # 스트림에 쓰기
                    is_playing = True
                else:
                    print("수신된 오디오 데이터 없음 (빈 청크)")
                audio_queue.task_done()

            except asyncio.QueueEmpty:
                if not is_playing and audio_queue.empty():
                    time.sleep(0.01)
                    continue # 큐가 비었지만, 재생 중이 아닐 때는 계속 확인
            except sd.PortAudioError as e:
                print(f"PortAudio 에러: {e}")
                is_playing = False
                break
            except Exception as e:
                print(f"오디오 재생 중 에러 발생: {e}")
                is_playing = False
                break

    except sd.PortAudioError as stream_error:
        print(f"스트림 생성 에러: {stream_error}") # 스트림 생성 에러 처리
    except Exception as e_stream_init:
        print(f"스트림 초기화 중 에러 발생: {e_stream_init}")
    finally:
        if stream and stream.active:
            stream.stop()
            stream.close() # 스트림 닫기 (자원 해제)
            print("오디오 스트림 중지 및 닫힘")
        is_playing = False
        print("오디오 재생 쓰레드 종료")


async def process_response(session):
    """ Gemini response를 비동기적으로 수신하고 오디오 큐에 넣는 task """
    global response_task # response_task를 global 변수로 사용
    print('start process_response')
    while response_task is not None:
        async for message in session.receive():
            if message.server_content.model_turn:
                for part in message.server_content.model_turn.parts:
                    if part.inline_data:
                        audio_chunk_bytes = part.inline_data.data # bytes 데이터
                        await audio_queue.put(audio_chunk_bytes)

            if message.server_content.turn_complete:
                print('turn complete received.')
                print(message.server_content)
                # await audio_queue.put(None) # 종료 신호
                # response_task = None # turn_complete 후 task 변수 초기화
    print('stop process_response')


def input_worker():
    """ 사용자 입력을 blocking 방식으로 받는 워커 쓰레드 함수 """
    while True:
        message = input("User> ")
        input_queue.put(message) # 입력 큐에 메시지 저장
        if message.lower() == "exit":
            break
    print("입력 쓰레드 종료")


async def main():
    global playback_thread, is_playing, response_task, input_thread

    if playback_thread is None or not playback_thread.is_alive(): # Ensure thread starts only once
        playback_thread = threading.Thread(target=playback_worker, daemon=True)
        playback_thread.start()
        print("새로운 재생 쓰레드 시작")

    if input_thread is None or not input_thread.is_alive():
        input_thread = threading.Thread(target=input_worker, daemon=True)
        input_thread.start()
        print("새로운 입력 쓰레드 시작")


    async with client.aio.live.connect(model=model_id, config=config) as session:
        print("Gemini Live API 연결됨. 'exit'를 입력하여 종료.")
        response_task = asyncio.create_task(process_response(session)) # task 시작 및 변수에 저장

        while True:
            message = await asyncio.get_running_loop().run_in_executor(None, input_queue.get) # non-blocking get from input_queue
            if message.lower() == "exit":
                break
            await session.send(input=message, end_of_turn=True)


    print("exit command received. waiting for response task to complete...") # 종료 대기 메시지

    if response_task is not None:
        response_task = None
        #await response_task # process_response task가 완료될 때까지 기다림
        print("response task completed.")

    if playback_thread and playback_thread.is_alive():
        await audio_queue.put(None) # 마지막 종료 신호
        await audio_queue.join()
        playback_thread.join(timeout=5)
        print("최종 재생 쓰레드 종료 대기 완료")

    if input_thread and input_thread.is_alive():
        input_queue.put("exit") # 입력 쓰레드 종료 신호 (혹시 input_queue.get() 에서 blocking 되어 있을 경우를 대비)
        input_thread.join(timeout=5) # 입력 쓰레드 join
        print("최종 입력 쓰레드 종료 대기 완료")


    print("프로그램 종료")


if __name__ == "__main__":
    asyncio.run(main())
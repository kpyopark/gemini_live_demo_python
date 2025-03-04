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

load_dotenv()

GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")


client = genai.Client(api_key=GEMINI_API_KEY, http_options={'api_version': 'v1alpha'})
model_id = "gemini-2.0-flash-exp"
config = {"responseModalities": ["TEXT"]}

SAMPLE_RATE=24000
DTYPE = np.int16  # Define the data type for audio
CHANNELS = 1 # Mono audio

audio_queue = asyncio.Queue() # 오디오 청크(bytes)를 저장할 큐
is_playing = False
playback_thread = None


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
                print(f"큐에서 오디오 청크 가져옴, 크기: {audio_chunk_np.shape}, 데이터 타입: {audio_chunk_np.dtype}") # 디버깅 출력

                if audio_chunk_np.size > 0:
                    print(f"재생 데이터 크기: {audio_chunk_np.size}, 샘플 레이트: {SAMPLE_RATE}") # 디버깅 출력
                    stream.write(audio_chunk_np) # 스트림에 쓰기
                    is_playing = True
                    print("오디오 청크 스트림에 쓰기")
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
        if stream and stream.active: # Changed from stream.is_active to stream.active
            stream.stop()
            stream.close() # 스트림 닫기 (자원 해제)
            print("오디오 스트림 중지 및 닫힘")
        is_playing = False
        print("오디오 재생 쓰레드 종료")


async def main():
    global playback_thread, is_playing

    async with client.aio.live.connect(model=model_id, config=config) as session:
        print("Gemini Live API 연결됨. 'exit'를 입력하여 종료.")

        while True:
            message = input("User> ")
            if message.lower() == "exit":
                break

            await session.send(input=message, end_of_turn=True)

            is_playing = False
            playback_thread = threading.Thread(target=playback_worker, daemon=True)
            playback_thread.start()
            print("새로운 재생 쓰레드 시작")


            async for message in session.receive():
                if message.server_content.model_turn:
                    for part in message.server_content.model_turn.parts:
                        if part.inline_data:
                            audio_chunk_bytes = part.inline_data.data # bytes 데이터
                            print(f"오디오 데이터 수신됨, 크기: {len(audio_chunk_bytes)} bytes") # 디버깅 출력
                            await audio_queue.put(audio_chunk_bytes)

                if message.server_content.turn_complete:
                    print('turn complete received.')
                    await audio_queue.put(None) # 종료 신호

    if playback_thread and playback_thread.is_alive():
        await audio_queue.put(None) # 마지막 종료 신호
        await audio_queue.join()
        playback_thread.join(timeout=5)
        print("최종 재생 쓰레드 종료 대기 완료")

    print("프로그램 종료")


if __name__ == "__main__":
    asyncio.run(main())
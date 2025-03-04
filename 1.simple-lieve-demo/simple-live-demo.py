import asyncio
from google import genai
from dotenv import load_dotenv
import numpy as np
import os
import io
import wave
import sounddevice as sd

load_dotenv()

GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")


client = genai.Client(api_key=GEMINI_API_KEY, http_options={'api_version': 'v1alpha'})
model_id = "gemini-2.0-flash-exp"
config = {"responseModalities": ["TEXT"]}

SAMPLE_RATE=24000

async def play_audio_stream(audio_stream_chunks):
    """오디오 스트림 청크를 재생합니다."""
    try:
        # 오디오 청크들을 numpy 배열로 합치기
        audio_data = np.concatenate(audio_stream_chunks, axis=0) if audio_stream_chunks else np.array([], dtype=DTYPE)

        if audio_data.size > 0: # Check if there is audio data to play
            sd.play(audio_data, samplerate=SAMPLE_RATE)
            sd.wait() # Wait until playback is finished
            print("오디오 재생 완료")
        else:
            print("수신된 오디오 데이터 없음")

    except sd.PortAudioError as e:
        print(f"PortAudio 에러: {e}")
    except Exception as e:
        print(f"오디오 재생 중 에러 발생: {e}")


async def main():
    async with client.aio.live.connect(model=model_id, config=config) as session:
        while True:
            message = input("User> ")
            if message.lower() == "exit":
                break
            await session.send(input=message, end_of_turn=True)

            audio_data = []
            async for message in session.receive():
                if message.server_content.model_turn:
                    for part in message.server_content.model_turn.parts:
                        if part.inline_data:
                            print('audio data received.')
                            audio_data.append(
                                np.frombuffer(part.inline_data.data, dtype=np.int16)
                            )
                if message.server_content.turn_complete:
                    print('audio played.')
                    await play_audio_stream(audio_data)

if __name__ == "__main__":
    asyncio.run(main())
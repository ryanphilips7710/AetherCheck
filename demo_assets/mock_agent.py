import asyncio
import json
import base64
import wave
import sys
import websockets

BACKEND_WS_URL = "ws://localhost:8000/ws/stream"

async def stream_audio_file(file_path: str, source_label: str = "remote"):
    print(f"[*] Opening audio file: {file_path}")
    try:
        wf = wave.open(file_path, "rb")
    except Exception as e:
        print(f"[!] Error opening file: {e}")
        return

    # Strict parameter check
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
        print("[!] Format error: File must be 16kHz, 16-bit, Mono PCM.")
        return

    print(f"[+] Connecting to {BACKEND_WS_URL}...")
    try:
        async with websockets.connect(BACKEND_WS_URL) as ws:
            print(f"[+] Connected! Streaming audio as '{source_label}'...")

            # 4096 samples = 256ms of audio at 16000 Hz
            chunk_size = 4096

            while True:
                data = wf.readframes(chunk_size)
                if not data:
                    print("[*] Reached end of audio stream.")
                    break

                b64_payload = base64.b64encode(data).decode("utf-8")
                message = {
                    "source": source_label,
                    "timestamp_ms": int(asyncio.get_event_loop().time() * 1000),
                    "audio_data": b64_payload
                }

                await ws.send(json.dumps(message))

                # Non-blocking listen for risk scores broadcast by backend
                try:
                    response = await asyncio.wait_for(ws.recv(), timeout=0.01)
                    print(f"[Backend Feedback] -> {response}")
                except asyncio.TimeoutError:
                    pass

                # Maintain realistic streaming delay
                await asyncio.sleep(chunk_size / 16000.0)

    except Exception as e:
        print(f"[!] Connection failed: {e}")
        print("    Ensure Dev 3 has started the FastAPI server (`uvicorn main:app --reload`).")

if __name__ == "__main__":
    target_file = sys.argv[1] if len(sys.argv) > 1 else "demo_assets/test_scenarios/cxo_fraud_attack.wav"
    asyncio.run(stream_audio_file(target_file))
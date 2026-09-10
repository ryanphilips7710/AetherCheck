import numpy as np
import torch
from silero_vad import load_silero_vad

class StreamVADState:
    def __init__(self, model, threshold=0.5):
        self.model = model
        self.threshold = threshold
        self.buffer = np.array([], dtype=np.float32)
        self.is_speaking = False
        self.last_speech_start_ts = None
        self.last_speech_end_ts = None
        self.chunk_size = 512

    def process_chunk(self, pcm_data, timestamp_ms):
        self.buffer = np.concatenate((self.buffer, pcm_data))
        event = None

        while len(self.buffer) >= self.chunk_size:
            frame = self.buffer[:self.chunk_size]
            self.buffer = self.buffer[self.chunk_size:]
            tensor_frame = torch.from_numpy(frame)
            
            speech_prob = self.model(tensor_frame, 16000).item()

            if speech_prob >= self.threshold and not self.is_speaking:
                self.is_speaking = True
                self.last_speech_start_ts = timestamp_ms
                event = "SPEECH_START"
            elif speech_prob < (self.threshold - 0.15) and self.is_speaking:
                self.is_speaking = False
                self.last_speech_end_ts = timestamp_ms
                event = "SPEECH_END"

        return event

class ConversationalPhysicsEngine:
    def __init__(self):
        # Load the Silero VAD model
        self.model = load_silero_vad()
        self.local_vad = StreamVADState(self.model)
        self.remote_vad = StreamVADState(self.model)

    def ingest_pcm(self, source, pcm_bytes, timestamp_ms):
        # Convert raw bytes to float32 audio
        audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0

        event = None
        if source == "local":
            event = self.local_vad.process_chunk(audio_float32, timestamp_ms)
        elif source == "remote":
            event = self.remote_vad.process_chunk(audio_float32, timestamp_ms)

        result = {"source": source, "event": event, "latency_ms": None, "flag": "NORMAL"}

        # Calculate floor-to-floor latency
        if source == "remote" and event == "SPEECH_START":
            if self.local_vad.last_speech_end_ts is not None:
                delta_t = timestamp_ms - self.local_vad.last_speech_end_ts
                
                # Math: \Delta t = T_{remote\_start} - T_{local\_end}
                if 50.0 <= delta_t <= 5000.0:
                    result["latency_ms"] = round(delta_t, 1)
                    
                    # AI models typically take > 750ms to respond
                    if delta_t >= 750.0:
                        result["flag"] = "AI_PIPELINE_DELAY_SUSPECTED"

        return result
import numpy as np
from physics_engine import ConversationalPhysicsEngine

def generate_fake_audio():
    # Generates 1 second of a flat tone
    t = np.linspace(0, 1.0, 16000, False)
    audio = np.sin(2 * np.pi * 440 * t) * 32767
    return audio.astype(np.int16).tobytes()

print("1. Booting Physics Engine (Will download Silero VAD if first run)...")
engine = ConversationalPhysicsEngine()

print("\n2. Simulating User (Local) speaking...")
fake_audio = generate_fake_audio()
engine.ingest_pcm("local", fake_audio, 1000.0)

print("3. User stops speaking at timestamp 2000ms...")
engine.ingest_pcm("local", b'\x00' * 1024, 2000.0) 

print("4. Simulating AI Caller (Remote) responding at timestamp 3300ms...")
print("   (This is a 1300ms gap, structurally impossible for a fluent human)")
result = engine.ingest_pcm("remote", fake_audio, 3300.0)

print("\n--- FINAL SCORING RESULT ---")
print(result)
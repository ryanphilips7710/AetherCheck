import sys
import os
import librosa
import soundfile as sf

def convert_mp3_to_16k_mono_wav(input_file: str, output_file: str):
    """
    Converts an audio file (MP3/WAV/M4A) to 16 kHz Mono 16-bit Linear PCM WAV.
    Required format for Silero-VAD and Faster-Whisper real-time engines.
    """
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return

    print(f"[*] Processing audio file: {input_file}")

    # librosa.load automatically resamples to 16000 Hz and collapses channels to mono
    audio_data, sample_rate = librosa.load(input_file, sr=16000, mono=True)

    # Ensure the destination directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    # Write as standard uncompressed 16-bit PCM WAV
    sf.write(output_file, audio_data, sample_rate, subtype="PCM_16")

    print(f"[+] Conversion complete!")
    print(f"    Saved as: {output_file}")
    print(f"    Sample Rate: {sample_rate} Hz")
    print(f"    Channels: Mono (1)")
    print(f"    Bit Depth: 16-bit PCM")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        # Default fallback paths for testing
        src = "demo_assets/test_scenarios/Natural_Conversation.mp3"
        dst = "demo_assets/test_scenarios/Natural_Conversation.wav"
        print(f"No arguments provided. Running default:\n  {src} -> {dst}\n")
    else:
        src = sys.argv[1]
        dst = sys.argv[2]

    convert_mp3_to_16k_mono_wav(src, dst)
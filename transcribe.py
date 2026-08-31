import os
import json
import wave
import argparse
from vosk import Model, KaldiRecognizer
from pydub import AudioSegment

# Configuration
MODEL_PATH = "./vosk-model-small-en-us-0.15"  # Change if you use a different model
TARGET_SAMPLE_RATE = 16000

def prepare_audio(input_path: str) -> str:
    """
    Converts any audio file to 16kHz, mono, 16-bit PCM WAV format required by Vosk.
    Returns the path to the converted temporary file.
    """
    print(f"Loading audio: {input_path}")
    audio = AudioSegment.from_file(input_path)
    
    # Convert to mono, 16kHz, 16-bit
    audio = audio.set_channels(1).set_frame_rate(TARGET_SAMPLE_RATE).set_sample_width(2)
    
    temp_output = "temp_converted.wav"
    audio.export(temp_output, format="wav")
    print(f"Audio prepared and saved to: {temp_output}")
    return temp_output

def transcribe_audio(audio_path: str):
    """
    Runs the prepared audio through the Vosk model and returns the transcribed text.
    """
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at '{MODEL_PATH}'")
        print("Please download a model from https://alphacephei.com/vosk/models")
        print("Example: wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip && unzip it here.")
        return

    print("Loading Vosk model... (this may take a few seconds)")
    model = Model(MODEL_PATH)

    wf = wave.open(audio_path, "rb")
    recognizer = KaldiRecognizer(model, wf.getframerate())

    print("Transcribing...\n" + "-" * 40)
    
    full_text = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            if result.get("text"):
                full_text.append(result["text"])
                print(f"→ {result['text']}")

    # Get any remaining speech at the end of the file
    final_result = json.loads(recognizer.FinalResult())
    if final_result.get("text"):
        full_text.append(final_result["text"])
        print(f"→ {final_result['text']}")

    print("-" * 40)
    return " ".join(full_text)

def main():
    parser = argparse.ArgumentParser(description="Convert any audio recording to text using Vosk.")
    parser.add_argument("audio_file", type=str, help="Path to the audio file (mp3, wav, m4a, etc.)")
    args = parser.parse_args()

    if not os.path.exists(args.audio_file):
        print(f"❌ Error: File '{args.audio_file}' not found.")
        return

    try:
        # 1. Prepare the audio
        prepared_audio = prepare_audio(args.audio_file)
        
        # 2. Transcribe
        final_transcription = transcribe_audio(prepared_audio)
        
        print("\n✅ FINAL TRANSCRIPTION:")
        print(final_transcription)

        # 3. Cleanup temp file
        if os.path.exists(prepared_audio):
            os.remove(prepared_audio)
            
    except Exception as e:
        print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main()
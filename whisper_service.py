import argparse
import whisper

def transcribe_audio(audio_path: str, model_name: str) -> str:
# Load the model once so the app can reuse it for multiple audio files. 
    model = whisper.load_model(model_name)   
    # transcribe the audio file and return the text
    result = model.transcribe(audio_path)
    return result["text"].strip()


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe an audio file with Whisper."
    )
    parser.add_argument("audio_file", help="Path to an audio file")
    parser.add_argument(
        "model_name",
        nargs="?",
        default="small",
        choices = ("tiny", "base", "small", "medium", "large", "turbo"),
    )
    args = parser.parse_args()

    print("\nTRANSCRIPTION:")
    print(transcribe_audio(args.audio_file, args.model_name))


if __name__ == "__main__":
    main()
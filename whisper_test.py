import whisper

model = whisper.load_model("small")
result = model.transcribe("test3.aac")
print(result["text"])
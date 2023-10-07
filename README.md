Speaker Diarization app that also has transcribing and AI Chat features.

The following code is an application to perform speech diarization (the process of separating an audio stream into segments according to speaker identity) and transcription (the process of translating speech into written text). It uses both PyAnnote and Whisper APIs, and can process audio either uploaded from a local file or fetched from a YouTube video URL.

# References
  - [pyannote.audio](https://github.com/pyannote/pyannote-audio)
  - [HuggingFace pyannote diarization](https://huggingface.co/pyannote/speaker-diarization-3.0)
  - [Whisper API](https://platform.openai.com/docs/guides/speech-to-text/quickstart)
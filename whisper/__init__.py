import os
from whisper.silero_vad.utils_vad import OnnxWrapper
from whisper.model import load_model


silero_model = OnnxWrapper(os.path.join(os.path.dirname(__file__), "assets", "silero_vad.onnx"))


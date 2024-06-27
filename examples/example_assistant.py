import numpy as np
import pyaudio
import whisper
import logging
import sounddevice as sd
from sshkeyboard import listen_keyboard, stop_listening

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

INPUT_DEFAULT_DURATION_SECONDS = 5
INPUT_FORMAT = pyaudio.paInt16
INPUT_CHANNELS = 1
INPUT_RATE = 16000
INPUT_CHUNK = 16000*5

##change this to your own device should you need
INPUT_DEVICE = sd.default.device[0]
logging.info(f"Using default device {INPUT_DEVICE}")

class Assistant:
    def __init__(self):
  
        self.audio = pyaudio.PyAudio()

        try:
            self.audio_stream = self.audio.open(format=INPUT_FORMAT,
                            channels=INPUT_CHANNELS,
                            rate=INPUT_RATE,
                            input=True,
                            start=True,
                            input_device_index=INPUT_DEVICE,
                            )
        except Exception as e:                                            
            logging.error(f"Error opening audio stream: {str(e)}")
            self.wait_exit()

        args = {"language": 'English',
        "name": "small.en",
        "precision": "fp32",
        "disable_cupy": False}

        self.temperature = tuple(np.arange(0, 1.0 + 1e-6, 0.2))
        self.decode_options = {"language": 'English',
                "disable_cupy": False}

        self.model = whisper.load_model(trt=False, **args)

        self.context = []
        self.frames = []
        self.released=False
        self.terminate=False

    def get_pressed(self, key):
        self.key = key
    
        if self.key == 'space':
            while not self.terminate:
                data = self.audio_stream.read(1024)
        
                self.frames.append(data)

    def get_release(self, key):

        if self.key == 'space':
           
            self.released = True
            self.terminate=True
        
            stop_listening()
            
    def shutdown(self):
        logging.info("Shutting down Assistant")
        self.audio.terminate()
        sys.exit()

    def waveform_from_mic(self) -> np.ndarray:
  
        result = self.frames
        result = np.frombuffer(b''.join(result), np.int16).astype(np.float32) * (1 / 32768.0)
        return result

    def main(self):
       
        while True:

            logging.info("hold space to begin")
            
            listen_keyboard(on_press=self.get_pressed, 
                            on_release=self.get_release)

            if self.released:

                speech = self.waveform_from_mic()
                result = self.model.transcribe(
                    speech, 
                    temperature=self.temperature,
                    **self.decode_options
                    )

                print('\nTRANSCRIPTION:\n')
                print(result['text'])
                print('\n')
                        
                self.frames = []
                self.release=False
                self.terminate=False

            elif ass.key == 'q':
                logging.info("Quit key pressed")
                ass.shutdown()


if __name__ == "__main__":

    logging.info("Starting Assistant...\n")
 
    ass = Assistant()
    ass.main()

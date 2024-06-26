import io
import sys
import os
from tqdm.auto import tqdm
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Union, TYPE_CHECKING
import numpy as np
import requests
import onnx
import warnings
import onnxruntime as ort

from whisper.utils import onnx_dtype_to_np_dtype_convert
from whisper.audio import SAMPLE_RATE, N_FRAMES, HOP_LENGTH, pad_or_trim, log_mel_spectrogram
from whisper.decoding import detect_language as detect_language_function, decode as decode_function, DecodingOptions, DecodingResult
from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE, get_tokenizer
from whisper.utils import exact_div, format_timestamp, optional_int, optional_float, str2bool, write_txt, write_vtt, write_srt

if TYPE_CHECKING:
    from whisper.model import Whisper

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)


_MODELS = {
    "tiny.en": "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/381_Whisper/pt/tiny.en.pt",
    "tiny": "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/381_Whisper/pt/tiny.pt",
    "base.en": "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/381_Whisper/pt/base.en.pt",
    "base": "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/381_Whisper/pt/base.pt",
    "small.en": "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/381_Whisper/pt/small.en.pt",
    "small": "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/381_Whisper/pt/small.pt",
    "medium.en": "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/381_Whisper/pt/medium.en.pt",
    "medium": "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/381_Whisper/pt/medium.pt",
    "large-v1": "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/381_Whisper/pt/large-v1.pt",
    "large-v2": "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/381_Whisper/pt/large-v2.pt",
}

def model_download(name: str, onnx_file_save_path: str='.') -> onnx.ModelProto:
    onnx_file = f'{name}.onnx'
    onnx_file_path = f'{onnx_file_save_path}/{onnx_file}'
    onnx_serialized_graph = None
    if not os.path.exists(onnx_file_path):
        try:
            url = f'https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/381_Whisper/onnx/whisper-onnx-xxx/float16/layer_fused_optimization_float16/{onnx_file}'
            onnx_serialized_graph = requests.get(url).content
            with io.BytesIO(onnx_serialized_graph) as f:
                onnx_graph: onnx.ModelProto = onnx.load(f)
                onnx.save(onnx_graph, f'{onnx_file_path}')
        except:
            onnx_file = f'{name}.onnx'
            onnx_file_path = f'{onnx_file_save_path}/{onnx_file}'
            if not os.path.exists(onnx_file_path):
                url = f'https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/381_Whisper/onnx/whisper-onnx-xxx/float16/no_optimization/{onnx_file}'
                onnx_serialized_graph = requests.get(url).content
                with io.BytesIO(onnx_serialized_graph) as f:
                    onnx_graph: onnx.ModelProto = onnx.load(f)
                    onnx.save(onnx_graph, f'{onnx_file_path}')
            else:
                onnx_graph: onnx.ModelProto = onnx.load(onnx_file_path)
                onnx_serialized_graph = onnx._serialize(onnx_graph)
    else:
        onnx_graph: onnx.ModelProto = onnx.load(onnx_file_path)
        onnx_serialized_graph = onnx._serialize(onnx_graph)
    return onnx_serialized_graph

def load_model(**decode_options):
    """
    Load a Whisper ASR model

    Parameters
    ----------
    name : str
        one of the official model names listed by `whisper.available_models()`, or
        path to a model checkpoint containing the model dimensions and the model state_dict.

    Returns
    -------
    model : Whisper
        The Whisper ASR model instance
    """

    name = decode_options['name']

    if name == "tiny":
        dims_config = {'n_mels': 80, 'n_vocab': 51865, 'n_audio_ctx': 1500, 'n_audio_state': 384, 'n_audio_head': 6, 'n_audio_layer': 4, 'n_text_ctx': 448, 'n_text_state': 384, 'n_text_head': 6, 'n_text_layer': 4}
    elif name == "tiny.en":
        dims_config = {'n_mels': 80, 'n_vocab': 51864, 'n_audio_ctx': 1500, 'n_audio_state': 384, 'n_audio_head': 6, 'n_audio_layer': 4, 'n_text_ctx': 448, 'n_text_state': 384, 'n_text_head': 6, 'n_text_layer': 4}
    elif name == "base":
        dims_config = {'n_mels': 80, 'n_vocab': 51865, 'n_audio_ctx': 1500, 'n_audio_state': 512, 'n_audio_head': 8, 'n_audio_layer': 6, 'n_text_ctx': 448, 'n_text_state': 512, 'n_text_head': 8, 'n_text_layer': 6}
    elif name == "base.en":
        dims_config = {'n_mels': 80, 'n_vocab': 51864, 'n_audio_ctx': 1500, 'n_audio_state': 512, 'n_audio_head': 8, 'n_audio_layer': 6, 'n_text_ctx': 448, 'n_text_state': 512, 'n_text_head': 8, 'n_text_layer': 6}
    elif name == "small":
        dims_config = {'n_mels': 80, 'n_vocab': 51865, 'n_audio_ctx': 1500, 'n_audio_state': 768, 'n_audio_head': 12, 'n_audio_layer': 12, 'n_text_ctx': 448, 'n_text_state': 768, 'n_text_head': 12, 'n_text_layer': 12}
    elif name == "small.en":
        dims_config = {'n_mels': 80, 'n_vocab': 51864, 'n_audio_ctx': 1500, 'n_audio_state': 768, 'n_audio_head': 12, 'n_audio_layer': 12, 'n_text_ctx': 448, 'n_text_state': 768, 'n_text_head': 12, 'n_text_layer': 12}
    elif name == "medium":
        dims_config = {'n_mels': 80, 'n_vocab': 51865, 'n_audio_ctx': 1500, 'n_audio_state': 1024, 'n_audio_head': 16, 'n_audio_layer': 24, 'n_text_ctx': 448, 'n_text_state': 1024, 'n_text_head': 16, 'n_text_layer': 24}
    elif name == "medium.en":
        dims_config = {'n_mels': 80, 'n_vocab': 51864, 'n_audio_ctx': 1500, 'n_audio_state': 1024, 'n_audio_head': 16, 'n_audio_layer': 24, 'n_text_ctx': 448, 'n_text_state': 1024, 'n_text_head': 16, 'n_text_layer': 24}
    elif name == "large-v1":
        dims_config = {'n_mels': 80, 'n_vocab': 51865, 'n_audio_ctx': 1500, 'n_audio_state': 1280, 'n_audio_head': 20, 'n_audio_layer': 32, 'n_text_ctx': 448, 'n_text_state': 1280, 'n_text_head': 20, 'n_text_layer': 32}
    elif name == "large-v2":
        dims_config = {'n_mels': 80, 'n_vocab': 51865, 'n_audio_ctx': 1500, 'n_audio_state': 1280, 'n_audio_head': 20, 'n_audio_layer': 32, 'n_text_ctx': 448, 'n_text_state': 1280, 'n_text_head': 20, 'n_text_layer': 32}
    else:
        raise ValueError(f"model type {name} not supported")

    dims = ModelDimensions(**dims_config)
    model = Whisper(dims=dims, 
                    model_name=name, 
                    **decode_options)
    return model

def available_models() -> List[str]:
    """Returns the names of available models"""
    return list(_MODELS.keys())

@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


class OnnxAudioEncoder():
    def __init__(
        self,
        model: str,
    ):
        super().__init__()
        
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED


        self.sess = \
            ort.InferenceSession(
                path_or_bytes=model_download(name=f'{model}_encoder'),
                providers=[
                    'CUDAExecutionProvider',
                    'CPUExecutionProvider'
                ],
                sess_options=sess_options
            )
        self.inputs = {
            input.name: onnx_dtype_to_np_dtype_convert(input.type) \
                for input in self.sess.get_inputs()
        }

    def __call__(
        self,
        mel: np.ndarray
    ) -> np.ndarray:
        result: np.ndarray = \
            self.sess.run(
                output_names=[
                    "output",
                ],
                input_feed={
                    "mel": mel.astype(self.inputs["mel"]),
                }
            )[0]
        return result


class OnnxTextDecoder():
    def __init__(
        self,
        model: str,
    ):
        super().__init__()
        
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED


        self.sess = \
            ort.InferenceSession(
                path_or_bytes=model_download(name=f'{model}_decoder'),
                providers=[
                    'CUDAExecutionProvider',
                    'CPUExecutionProvider'
                ],
                sess_options=sess_options
            )
        self.inputs = {
            input.name: onnx_dtype_to_np_dtype_convert(input.type) \
                for input in self.sess.get_inputs()
        }

    def __call__(
        self,
        x: np.ndarray,
        xa: np.ndarray,
        kv_cache: np.ndarray,
        offset: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        outputs = \
            self.sess.run(
                output_names=[
                    "logits",
                    "output_kv_cache",
                    "cross_attention_qks",
                ],
                input_feed={
                    "tokens": x.astype(self.inputs["tokens"]),
                    "audio_features": xa.astype(self.inputs["audio_features"]),
                    "kv_cache": kv_cache.astype(self.inputs["kv_cache"]),
                    "offset": np.array([offset], dtype=self.inputs["offset"]),
                }
            )
        logits: np.ndarray = outputs[0]
        output_kv_cache: np.ndarray = outputs[1]
        cross_attention_qks: np.ndarray = outputs[2]
        return logits.astype(np.float32), output_kv_cache.astype(np.float32)


class Whisper():
    def __init__(
        self,
        dims: ModelDimensions,
        model_name: str,
        **decode_options

    ):
        super().__init__()
        self.model_name = model_name
        self.dims = dims
        self.encoder = OnnxAudioEncoder(model=model_name)
        self.decoder = OnnxTextDecoder(model=model_name)


        if decode_options.get("language", None) is None:
            if verbose:
                print("Detecting language using up to the first 30 seconds. Use `--language` to specify the language")
            segment = pad_or_trim(mel, N_FRAMES)
            _, probs = self.detect_language(segment)
            decode_options["language"] = max(probs, key=probs.get)
            if verbose is not None:
                print(f"Detected language: {LANGUAGES[decode_options['language']].title()}")

        self.language = decode_options["language"]
        task = decode_options.get("task", "transcribe")
        self.tokenizer = get_tokenizer(self.is_multilingual, language=self.language, task=task)


    def embed_audio(
        self,
        mel: np.ndarray,
    ):
        return self.encoder(mel)

    def logits(
        self,
        tokens: np.ndarray,
        audio_features: np.ndarray,
    ):
        kv_cache = self.new_kv_cache(tokens.shape[0], tokens.shape[-1])
        output, _ = self.decoder(tokens, audio_features, kv_cache=kv_cache, offset=0)
        return output

    def __call__(
        self,
        mel: np.ndarray,
        tokens: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        kv_cache = self.new_kv_cache(tokens.shape[0], tokens.shape[-1])
        output, _ = self.decoder(tokens, self.encoder(mel), kv_cache=kv_cache, offset=0)
        return output

    @property
    def is_multilingual(self):
        return self.dims.n_vocab == 51865

    def new_kv_cache(
        self,
        n_group: int,
        length: int,
    ):
        if self.model_name == "tiny.en" or self.model_name == "tiny":
            size = [8, n_group, length, 384]
        elif self.model_name == "base.en" or self.model_name == "base":
            size = [12, n_group, length, 512]
        elif self.model_name == "small.en" or self.model_name == "small":
            size = [24, n_group, length, 768]
        elif self.model_name == "medium.en" or self.model_name == "medium":
            size = [48, n_group, length, 1024]
        elif self.model_name == "large-v1" or self.model_name == "large-v2":
            size = [64, n_group, length, 1280]
        else:
            raise ValueError(f"Unsupported model type: {self.type}")
        return np.zeros(size, dtype=np.float16)

    
    def transcribe(self,
        audio: Union[str, np.ndarray],
        *,
        verbose: Optional[bool] = None,
        temperature: Union[float, Tuple[float, ...]] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        compression_ratio_threshold: Optional[float] = 2.4,
        logprob_threshold: Optional[float] = -1.0,
        no_speech_threshold: Optional[float] = 0.6,
        condition_on_previous_text: bool = True,
        **decode_options,
    ):
        """
        Transcribe an audio file using Whisper

        Parameters
        ----------
        model: Whisper
            The Whisper model instance

        audio: Union[str, np.ndarray]
            The path to the audio file to open, or the audio waveform

        verbose: bool
            Whether to display the text being decoded to the console. If True, displays all the details,
            If False, displays minimal details. If None, does not display anything

        temperature: Union[float, Tuple[float, ...]]
            Temperature for sampling. It can be a tuple of temperatures, which will be successfully used
            upon failures according to either `compression_ratio_threshold` or `logprob_threshold`.

        compression_ratio_threshold: float
            If the gzip compression ratio is above this value, treat as failed

        logprob_threshold: float
            If the average log probability over sampled tokens is below this value, treat as failed

        no_speech_threshold: float
            If the no_speech probability is higher than this value AND the average log probability
            over sampled tokens is below `logprob_threshold`, consider the segment as silent

        condition_on_previous_text: bool
            if True, the previous output of the model is provided as a prompt for the next window;
            disabling may make the text inconsistent across windows, but the model becomes less prone to
            getting stuck in a failure loop, such as repetition looping or timestamps going out of sync.

        decode_options: dict
            Keyword arguments to construct `DecodingOptions` instances

        Returns
        -------
        A dictionary containing the resulting text ("text") and segment-level details ("segments"), and
        the spoken language ("language"), which is detected when `decode_options["language"]` is None.
        """

        mel: np.ndarray = log_mel_spectrogram(audio, decode_options.pop("disable_cupy"))
        mel = mel[np.newaxis, ...]
        
        def decode_with_fallback(segment: np.ndarray) -> List[DecodingResult]:

            print('')
            temperatures = [temperature] if isinstance(temperature, (int, float)) else temperature
            kwargs = {**decode_options}
            t = temperatures[0]
            if t == 0:
                best_of = kwargs.pop("best_of", None)
            else:
                best_of = kwargs.get("best_of", None)

            options = DecodingOptions(**kwargs, temperature=t)
            results = self.decode(segment, options)

            kwargs.pop("beam_size", None)  # no beam search for t > 0
            kwargs.pop("patience", None)  # no patience for t > 0
            kwargs["best_of"] = best_of  # enable best_of for t > 0
            for t in temperatures[1:]:
                needs_fallback = [
                    compression_ratio_threshold is not None
                    and result.compression_ratio > compression_ratio_threshold
                    or logprob_threshold is not None
                    and result.avg_logprob < logprob_threshold
                    for result in results
                ]
                if any(needs_fallback):
                    options = DecodingOptions(**kwargs, temperature=t)
                    retries = self.decode(segment[needs_fallback], options)
                    for retry_index, original_index in enumerate(np.nonzero(needs_fallback)[0]):
                        results[original_index] = retries[retry_index]

            return results

        seek = 0
        input_stride = exact_div(
            N_FRAMES, self.dims.n_audio_ctx
        )  # mel frames per output token: 2
        time_precision = (
            input_stride * HOP_LENGTH / SAMPLE_RATE
        )  # time per output token: 0.02 (seconds)
        all_tokens = []
        all_segments = []
        prompt_reset_since = 0

        initial_prompt = decode_options.pop("initial_prompt", None) or []
        if initial_prompt:
            initial_prompt = self.tokenizer.encode(" " + initial_prompt.strip())
            all_tokens.extend(initial_prompt)

        def add_segment(
            *, start: float, end: float, text_tokens: np.ndarray, result: DecodingResult
        ):
            text = self.tokenizer.decode([token for token in text_tokens if token < self.tokenizer.eot])
            if len(text.strip()) == 0:  # skip empty text output
                return

            all_segments.append(
                {
                    "id": len(all_segments),
                    "seek": seek,
                    "start": start,
                    "end": end,
                    "text": text,
                    "tokens": result.tokens,
                    "temperature": result.temperature,
                    "avg_logprob": result.avg_logprob,
                    "compression_ratio": result.compression_ratio,
                    "no_speech_prob": result.no_speech_prob,
                }
            )
            if verbose:
                print(f"[{format_timestamp(start)} --> {format_timestamp(end)}] {text}", flush=True)

        # show the progress bar when verbose is False (otherwise the transcribed text will be printed)
        num_frames = mel.shape[-1]
        previous_seek_value = seek

        with tqdm(total=num_frames, unit='frames', disable=verbose is not False, miniters=1) as pbar:
            while seek < num_frames:
                timestamp_offset = float(seek * HOP_LENGTH / SAMPLE_RATE)
                segment = pad_or_trim(mel[:, :, seek:], N_FRAMES)
                segment_duration = segment.shape[-1] * HOP_LENGTH / SAMPLE_RATE

                decode_options["prompt"] = all_tokens[prompt_reset_since:]
                result = decode_with_fallback(segment)[0]
                tokens = result.tokens

                if no_speech_threshold is not None:
                    # no voice activity check
                    should_skip = result.no_speech_prob > no_speech_threshold
                    if logprob_threshold is not None and result.avg_logprob > logprob_threshold:
                        # don't skip if the logprob is high enough, despite the no_speech_prob
                        should_skip = False

                    if should_skip:
                        seek += segment.shape[-1]  # fast-forward to the next segment boundary
                        continue

                timestamp_tokens: np.ndarray = np.greater_equal(tokens, self.tokenizer.timestamp_begin)
                consecutive = np.add(np.where(timestamp_tokens[:-1] & timestamp_tokens[1:])[0], 1)
                if len(consecutive) > 0:  # if the output contains two consecutive timestamp tokens
                    last_slice = 0
                    for current_slice in consecutive:
                        sliced_tokens = tokens[last_slice:current_slice]
                        start_timestamp_position = (
                            sliced_tokens[0] - self.tokenizer.timestamp_begin
                        )
                        end_timestamp_position = (
                            sliced_tokens[-1] - self.tokenizer.timestamp_begin
                        )
                        add_segment(
                            start=timestamp_offset + start_timestamp_position * time_precision,
                            end=timestamp_offset + end_timestamp_position * time_precision,
                            text_tokens=sliced_tokens[1:-1],
                            result=result,
                        )
                        last_slice = current_slice
                    last_timestamp_position = (
                        tokens[last_slice - 1] - self.tokenizer.timestamp_begin
                    )
                    seek += last_timestamp_position * input_stride
                    all_tokens.extend([t for t in tokens[:last_slice + 1]])
                else:
                    duration = segment_duration
                    tokens = np.asarray(tokens) if isinstance(tokens, list) else tokens
                    timestamps = tokens[
                        np.ravel_multi_index(np.nonzero(timestamp_tokens), timestamp_tokens.shape)
                    ]
                    if len(timestamps) > 0:
                        # no consecutive timestamps but it has a timestamp; use the last one.
                        # single timestamp at the end means no speech after the last timestamp.
                        last_timestamp_position = timestamps[-1] - self.tokenizer.timestamp_begin
                        duration = last_timestamp_position * time_precision

                    add_segment(
                        start=timestamp_offset,
                        end=timestamp_offset + duration,
                        text_tokens=tokens,
                        result=result,
                    )

                    seek += segment.shape[-1]
                    all_tokens.extend(list(tokens))

                if not condition_on_previous_text or result.temperature > 0.5:
                    # do not feed the prompt tokens if a high temperature was used
                    prompt_reset_since = len(all_tokens)

                # update progress bar
                pbar.update(min(num_frames, seek) - previous_seek_value)
                previous_seek_value = seek

        return dict(text=self.tokenizer.decode(all_tokens[len(initial_prompt):]), segments=all_segments, language=self.language)

    detect_language = detect_language_function
    decode = decode_function

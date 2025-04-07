import os
import numpy as np
import soundfile as sf
import torch
from transformers import pipeline
from pydub import AudioSegment, utils as pydub_utils
import nltk
from nltk.tokenize import sent_tokenize
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt', quiet=True)
import re
import uuid
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import concurrent.futures
import openai
import gender_guesser.detector as gender
import requests

SCRIPT_FILE = "sample.txt"
OUTPUT_DIR = "Audio_Output_MusicGen_Formatted"
TARGET_SAMPLE_RATE = 24000
LOG_LEVEL = "INFO"
NARRATOR_NAME = "Narrator"
OPENAI_TTS_MODEL = "tts-1"
DEFAULT_VOICE = "alloy"
NARRATOR_VOICE = "shimmer"
OPENAI_MALE_VOICES = ["alloy", "echo", "onyx"]
OPENAI_FEMALE_VOICES = ["nova", "shimmer", "fable"]
TTS_SPEED = 1.0
MUSICGEN_MODEL = "facebook/musicgen-small"
MUSICGEN_TOKENS_PER_SECOND = 50
FALLBACK_MUSIC_PROMPT = "calm ambient background music"
MUSIC_VOLUME = 0.01
MUSIC_FADE_IN_SEC = 3.0
MUSIC_FADE_START_OFFSET_SEC = 1.0
MUSIC_FINAL_FADE_DURATION_SEC = 6.0
FREESOUND_SEARCH_TIMEOUT = 15
FREESOUND_DOWNLOAD_TIMEOUT = 45
SFX_TARGET_DURATION_SEC = 3.0
GENERATE_PLACEHOLDER_SFX = True
TIMELINE_START_PADDING_SEC = 0.2
TIMELINE_END_PADDING_SEC = 2.0
DEFAULT_PAUSE_SEC = 0.35
MIN_PAUSE_SEC = 0.15
SFX_VOLUME = 0.65
SPEECH_VOLUME = 1.0
DEFAULT_FADE_SEC = 0.02
SFX_FADE_IN_SEC = 0.05
SFX_FADE_OUT_SEC = 0.05
NORMALIZE_OUTPUT = True
NORMALIZATION_TARGET_DBFS = -16.0
OUTPUT_FORMAT = "wav"
MAX_WORKERS = 16

# --- Dataclasses ---
@dataclass
class ScriptSegment:
    id: str
    text: str
    segment_type: str
    speaker: Optional[str] = None
    emotion: Optional[str] = "neutral"
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    voice_file: Optional[str] = None
    duration: Optional[float] = None
    _temp_audio_object: Optional[AudioSegment] = field(default=None, repr=False)

@dataclass
class AudioElement:
    id: str
    audio_type: str
    file_path: str
    start_time: float
    duration: Optional[float] = None
    end_time: Optional[float] = None
    volume: float = 1.0
    fade_in: float = 0.0
    fade_out: float = 0.0
    effects: List[Dict] = field(default_factory=list)
    description: Optional[str] = None
    audio_object: Optional[AudioSegment] = None

@dataclass
class ProductionTimeline:
    segments: List[ScriptSegment]
    audio_elements: List[AudioElement]
    duration: float
    sample_rate: int

# --- Main Pipeline Class ---
class AudioPipeline:
    def __init__(self):
        self.output_dir = OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        self.setup_logging()

        self.freesound_api_key = os.getenv("FREESOUND_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.freesound_api_key:
            self.logger.warning("FREESOUND_API_KEY missing (SFX may fail).")
        if not self.openai_api_key:
            self.logger.warning("OPENAI_API_KEY missing (TTS will fail).")

        if torch.cuda.is_available():
            self.device = "cuda"
            self.logger.info("CUDA (GPU) available, using GPU for MusicGen.")
        else:
            self.device = "cpu"
            self.logger.info("CUDA not available, using CPU for MusicGen (will be slow).")

        self.openai_client = self.setup_openai_client()
        self.gender_detector = gender.Detector(case_sensitive=False)
        self.logger.info("Gender detector initialized.")
        self._generated_files_tracker = set()
        self.music_pipeline = None
        self.load_local_models()

    def setup_logging(self):
        log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
        logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('AudioPipeline')
        self.logger.info(f"Logging setup with level: {LOG_LEVEL.upper()}")

    def load_local_models(self):
        self.logger.info("Loading local models (MusicGen)...")
        try:
            device_index = 0 if self.device == "cuda" else -1
            self.music_pipeline = pipeline("text-to-audio", model = MUSICGEN_MODEL, device = device_index)
            self.logger.info(f"MusicGen model '{MUSICGEN_MODEL}' loaded on device '{self.device}'.")
        except ImportError:
             self.logger.error("ImportError loading MusicGen. Check 'transformers'/'torch'.", exc_info=True)
             self.music_pipeline = None
        except Exception as e:
            self.logger.error(f"Failed to load MusicGen model: {e}", exc_info=True)
            self.music_pipeline = None

    def setup_openai_client(self) -> Optional[openai.OpenAI]:
        if not self.openai_api_key:
            return None
        try:
            client = openai.OpenAI(api_key=self.openai_api_key)
            client.models.list() # Verify connection
            self.logger.info("OpenAI client initialized successfully.")
            return client
        except Exception as e:
            self.logger.error(f"OpenAI client init failed: {e}", exc_info=True)
            return None

    def identify_speaker_gender(self, name: str) -> str:
        if not name or not isinstance(name, str):
            return "unknown"
        first_name = name.split()[0]
        guess = "unknown"
        try:
            guess = self.gender_detector.get_gender(first_name)
            if guess.startswith("mostly_"):
                guess = guess.split("_")[1]
            if guess == "andy":
                guess = "unknown"
            return guess if guess in ["male", "female"] else "unknown"
        except Exception as e:
            self.logger.warning(f"Gender detection failed for '{name}': {e}. Returning 'unknown'.")
            return "unknown" # Return unknown on error

    def analyze_audio_file(self, audio_path: str) -> Dict:
        self.logger.debug(f"Analyzing: {audio_path}")
        duration = 0
        sr = 0
        try:
            info = sf.info(audio_path)
            duration = info.duration
            sr = info.samplerate
        except Exception as e_sf:
            self.logger.warning(f"Soundfile analysis failed for {audio_path}: {e_sf}. Trying pydub.")
            try:
                 audio = AudioSegment.from_file(audio_path)
                 duration = audio.duration_seconds
                 sr = audio.frame_rate
            except Exception as e_pd:
                 self.logger.error(f"Pydub analysis also failed for {audio_path}: {e_pd}.")
                 # Return dict with zeros if both fail
                 return {"duration": 0, "sample_rate": 0}
        return {"duration": duration, "sample_rate": sr}

    def cleanup_intermediate_files(self):
        self.logger.info(f"Cleaning up {len(self._generated_files_tracker)} intermediate files...")
        cleaned_count = 0
        # Iterate over a copy of the set as we modify it
        for file_path in list(self._generated_files_tracker):
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    self.logger.debug(f"Removed: {file_path}")
                    self._generated_files_tracker.remove(file_path)
                    cleaned_count += 1
                except OSError as e:
                    self.logger.warning(f"Could not remove {file_path}: {e}")
            elif file_path in self._generated_files_tracker:
                # Remove non-existent path from tracker if found
                self._generated_files_tracker.remove(file_path)
        self.logger.info(f"Removed {cleaned_count} files.")

    def process_script_to_audio(self, script_path: str):
        self.logger.info(f"Starting audio production for script: {script_path}")
        self._generated_files_tracker = set() # Reset tracker

        script_name = os.path.splitext(os.path.basename(script_path))[0]
        output_filename_base = os.path.join(self.output_dir, f"{script_name}_output")

        # Initialize components
        script_parser = ScriptParser(self)
        content_gen = ContentGenerator(self)
        synchronizer = Synchronizer(self)
        compiler = AudioCompiler(self)

        # Parse Script
        music_prompt_from_script, segments = script_parser.parse_script(script_path)
        if segments is None:
            self.logger.error(f"Script parsing failed: {script_path}.")
            return None
        self.logger.info(f"Parsed {len(segments)} segments.")
        if music_prompt_from_script:
            self.logger.info(f"Found background music prompt: '{music_prompt_from_script}'")
        else:
            self.logger.info("No background music prompt in script, using fallback.")

        # Identify segments needing generation
        segments_to_generate = []
        for seg in segments:
             if seg.segment_type not in ['dialogue', 'narration', 'sfx']:
                 continue
             if seg.voice_file and os.path.exists(seg.voice_file):
                 continue # Skip pre-set files
             segments_to_generate.append(seg)

        # Generate/Fetch Speech & SFX
        generated_audio_data: Dict[str, Dict] = {}
        if segments_to_generate:
            self.logger.info(f"Starting parallel tasks for {len(segments_to_generate)} speech/SFX segments...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_segment = {}
                # Submit tasks only if the required client/key exists
                for seg in segments_to_generate:
                    future = None
                    if seg.segment_type == 'sfx':
                        if self.freesound_api_key:
                            future = executor.submit(content_gen.generate_or_select_sfx, seg.text, SFX_TARGET_DURATION_SEC)
                        else:
                            generated_audio_data[seg.id] = {"error": "Freesound API key missing", "audio_object": None}
                    elif seg.segment_type in ['dialogue', 'narration']:
                        if self.openai_client:
                             future = executor.submit(content_gen.generate_speech, seg, self)
                        else:
                             generated_audio_data[seg.id] = {"error": "OpenAI client unavailable", "audio_object": None}
                    if future:
                         future_to_segment[future] = seg

                # Process results
                for future in concurrent.futures.as_completed(future_to_segment):
                    segment = future_to_segment[future]
                    try:
                        result = future.result()
                        generated_audio_data[segment.id] = result # Store result (could contain error)
                        if result and result.get("file_path"):
                            self.logger.info(f"Finished {segment.segment_type.upper()} task: {segment.id} -> {result.get('file_path')}")
                            self._generated_files_tracker.add(result["file_path"]) # Track successful generation
                        else:
                             self.logger.error(f"Task failed for {segment.id}: {result.get('error', 'Unknown error')}")
                    except Exception as exc:
                        self.logger.error(f"Task for segment {segment.id} threw exception: {exc}", exc_info=True)
                        generated_audio_data[segment.id] = {"error": str(exc), "audio_object": None} # Store error
            self.logger.info("Parallel speech/SFX tasks complete.")
        else:
             self.logger.info("No speech/SFX needed generation (or using pre-set files).")

        # Validate Segments & Prepare for Timeline
        valid_segments_for_timeline: List[ScriptSegment] = []
        music_result = None
        music_audio_object = None

        for segment in segments:
            if segment.segment_type == 'direction':
                continue

            audio_info = None
            audio_obj = None
            file_path = None
            duration = None
            is_preset = False

            # Check generated data first
            if segment.id in generated_audio_data:
                audio_info = generated_audio_data[segment.id]
                # Skip if generation failed
                if not audio_info or not audio_info.get("file_path") or audio_info.get("duration", 0) <= 0:
                    self.logger.warning(f"Excluding generated segment {segment.id}: {audio_info.get('error', 'N/A')}")
                    continue
                audio_obj = audio_info.get("audio_object")
                file_path = audio_info.get("file_path")
                duration = audio_info.get("duration")
            # Check for pre-set files second
            elif segment.voice_file and os.path.exists(segment.voice_file):
                is_preset = True
                analysis = self.analyze_audio_file(segment.voice_file)
                if analysis and analysis.get("duration", 0) > 0:
                     file_path = segment.voice_file
                     duration = analysis["duration"]
                     try:
                         # Load pre-set file into memory object
                         audio_obj = AudioSegment.from_file(file_path)
                         self.logger.debug(f"Loaded pre-set file object: {file_path}")
                     except Exception as load_e:
                         self.logger.error(f"Failed to load pre-set file {file_path}: {load_e}. Proceeding without object.")
                         audio_obj = None # Continue without object optimization
                else:
                    self.logger.warning(f"Excluding segment {segment.id}: Pre-set file analysis failed.")
                    continue
            # If no valid source, skip segment
            else:
                self.logger.warning(f"Excluding segment {segment.id}: No valid audio source found.")
                continue

            # Final check for valid path/duration
            if not file_path or not duration:
                self.logger.warning(f"Excluding segment {segment.id}: Missing file path or duration after checks.")
                continue

            # Update segment and store object temporarily
            segment.voice_file = file_path
            segment.duration = duration
            segment._temp_audio_object = audio_obj
            valid_segments_for_timeline.append(segment)

        # Build Content Timeline
        timeline = synchronizer.build_timeline(valid_segments_for_timeline, target_sr=TARGET_SAMPLE_RATE)
        if not timeline or timeline.duration <= 0:
            self.logger.error("Timeline build failed or content has zero duration.")
            return None
        self.logger.info(f"Content timeline built. Content duration: {timeline.duration:.2f}s")

        # Generate Background Music (MusicGen)
        music_prompt = music_prompt_from_script or FALLBACK_MUSIC_PROMPT
        if self.music_pipeline:
            self.logger.info(f"Generating music via MusicGen: '{music_prompt}'")
            # Calculate duration slightly longer than content + padding
            desired_music_duration = timeline.duration + MUSIC_FADE_START_OFFSET_SEC # Aim to cover content + fade offset
            if desired_music_duration <= 0:
                 self.logger.warning("Content duration zero, cannot generate music.")
            else:
                 music_result = content_gen.generate_music(music_prompt, duration=desired_music_duration)
                 if music_result and music_result.get("file_path"):
                     self.logger.info(f"Generated music: {music_result['file_path']} ({music_result['duration']:.2f}s)")
                     self._generated_files_tracker.add(music_result["file_path"])
                     music_audio_object = music_result.get("audio_object")
                     music_element = AudioElement(
                         id="background_music_musicgen", audio_type="music", file_path=music_result["file_path"],
                         start_time=0.0, duration=music_result["duration"], end_time=music_result["duration"],
                         volume=MUSIC_VOLUME, fade_in=MUSIC_FADE_IN_SEC, fade_out=MUSIC_FINAL_FADE_DURATION_SEC,
                         description=music_prompt, audio_object=music_audio_object
                     )
                     timeline.audio_elements.append(music_element)
                     # Update timeline total duration based on music length
                     timeline.duration = max(timeline.duration, music_result["duration"])
                     self.logger.info(f"Added MusicGen music. Total timeline duration: {timeline.duration:.2f}s")
                 else:
                     self.logger.error(f"MusicGen failed: {music_result.get('error', 'N/A')}")
        else:
            self.logger.warning("MusicGen pipeline unavailable, skipping background music.")

        # Adjust Timing & Compile
        adjusted_timeline = synchronizer.adjust_timing(timeline)
        output_path = f"{output_filename_base}.{OUTPUT_FORMAT}"
        final_output_path = compiler.compile_timeline(adjusted_timeline, output_path)

        # Cleanup
        if final_output_path:
             self.logger.info(f"Audio production finished: {final_output_path}")
             self.cleanup_intermediate_files()
        else:
             self.logger.error("Audio compilation failed.")
        return final_output_path


# --- Script Parser Class ---
class ScriptParser:
    def __init__(self, pipeline: AudioPipeline):
        self.pipeline = pipeline
        self.logger = pipeline.logger
        self.music_r = re.compile(r"^\s*Background Music\s*:\s*(?:['\"](.+?)['\"]|([^'\"\s].*?))\s*$", re.IGNORECASE)        
        self.sfx_r = re.compile(r"^(?:SFX|Sound effect)\s*[:\-]\s*(.+)|\[(?:Sound of|SFX)\s+([^\]]+)\]", re.I)
        self.spk_r = re.compile(r'^([\w\s\'\-]+):\s*(.*)')
        self.dir_r = re.compile(r'^[\(\[].*?[\)\]]$')

    def parse_script(self, script_path: str) -> Tuple[Optional[str], Optional[List[ScriptSegment]]]:
        self.logger.info(f"Parsing script: {script_path}")
        music_prompt = None
        segments: List[ScriptSegment] = []
        narrator_name = NARRATOR_NAME

        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                script_lines = f.readlines()
        except FileNotFoundError:
            self.logger.error(f"Script file not found: {script_path}")
            return None, None
        except Exception as e:
            self.logger.error(f"Error reading script: {e}", exc_info=True)
            return None, None

        for line_num, line in enumerate(script_lines, 1):
            line = line.strip()
            if not line:
                continue

            music_match = self.music_r.match(line)
            if music_match:
                if music_prompt is None:  # Only set the first occurrence
                    prompt_text = music_match.group(1) or music_match.group(2)
                    if prompt_text:
                        music_prompt = prompt_text.strip()
                        self.logger.info(f"Found music prompt line {line_num}: '{music_prompt}'")
                    else:
                        self.logger.warning(f"Music prompt matched on line {line_num}, but no text: '{line}'")
                else:
                    self.logger.debug(f"Skipping additional music prompt on line {line_num}: '{line}'")
                continue  # Always skip this line

            sfx_match = self.sfx_r.match(line)
            if sfx_match:
                sfx_desc = (sfx_match.group(1) or sfx_match.group(2) or "").strip()
                if sfx_desc:
                    seg_id = f"s_{line_num}_sfx_{uuid.uuid4().hex[:6]}"
                    segments.append(ScriptSegment(id=seg_id, text=sfx_desc, segment_type='sfx'))
                continue

            if self.dir_r.fullmatch(line):
                seg_id = f"s_{line_num}_dir_{uuid.uuid4().hex[:6]}"
                segments.append(ScriptSegment(id=seg_id, text=line, segment_type='direction'))
                continue

            try:
                sentences = sent_tokenize(line)
            except Exception:
                self.logger.warning(f"NLTK failed line {line_num}, using whole line.")
                sentences = [line]

            current_speaker = None  # Reset per line unless dialogue continues
            for sent_num, sentence in enumerate(sentences):
                sentence = sentence.strip()
                seg_id = f"s_{line_num}_{sent_num}_{uuid.uuid4().hex[:6]}"
                if not sentence:
                    continue

                speaker_match = self.spk_r.match(sentence)
                text_content = sentence
                segment_type = 'narration'
                speaker = narrator_name

                if speaker_match:
                    potential_speaker = speaker_match.group(1).strip()
                    potential_text = speaker_match.group(2).strip()
                    is_potential_tag = (len(potential_speaker.split()) < 4 and
                                        len(potential_speaker) < 30 and
                                        potential_speaker.replace(" ","").replace("'","").replace("-","").isalnum())
                    is_likely_start = potential_speaker.isupper() or potential_speaker.istitle()

                    if is_potential_tag and is_likely_start and potential_text and not self.dir_r.fullmatch(potential_text):
                        speaker = potential_speaker.title()
                        text_content = potential_text
                        segment_type = 'dialogue'
                        current_speaker = speaker
                    else:
                        if current_speaker:
                            segment_type = 'dialogue'
                            speaker = current_speaker
                        text_content = sentence

                else:
                    if current_speaker:
                        segment_type = 'dialogue'
                        speaker = current_speaker
                    text_content = sentence

                segments.append(ScriptSegment(id=seg_id, text=text_content, segment_type=segment_type, speaker=speaker))

        self.logger.info(f"Parsed into {len(segments)} content segments.")
        return music_prompt, segments

class ContentGenerator:
    def __init__(self, pipeline: AudioPipeline):
        self.pipeline = pipeline
        self.logger = pipeline.logger

    def generate_speech(self, segment: ScriptSegment, pipeline_instance: AudioPipeline) -> Dict:
            if not pipeline_instance.openai_client: 
                return {"error": "OpenAI client unavailable", "audio_object": None}
            if not segment.text: 
                return {"error": "No text", "audio_object": None}
            spk = segment.speaker 
            voice = DEFAULT_VOICE
            if spk == NARRATOR_NAME: 
                voice = NARRATOR_VOICE
            elif spk:
                gender = pipeline_instance.identify_speaker_gender(spk)
                voices = None
                if gender == "male": 
                    voices = OPENAI_MALE_VOICES
                elif gender == "female": 
                    voices = OPENAI_FEMALE_VOICES
                if voices: 
                    voice = voices[hash(spk) % len(voices)]
            path = os.path.join(self.pipeline.output_dir, f"voice_{segment.id}.wav") 
            audio_obj = None
            try:
                with pipeline_instance.openai_client.audio.speech.with_streaming_response.create(model=OPENAI_TTS_MODEL, voice=voice, input=segment.text, response_format="wav", speed=TTS_SPEED) as resp: 
                    resp.stream_to_file(path)
                if not os.path.exists(path) or os.path.getsize(path) == 0: 
                    raise ValueError("Output file empty/missing.")
                info = sf.info(path)
                duration = info.duration
                sr = info.samplerate
                try: 
                    audio_obj = AudioSegment.from_file(path)
                except Exception as e: 
                    self.logger.error(f"Generated {path} but failed load: {e}") 
                    return {"file_path": path, "duration": duration, "sample_rate": sr, "audio_object": None, "error": "Load fail"}
                return {"file_path": path, "duration": duration, "sample_rate": sr, "audio_object": audio_obj}
            except Exception as e: 
                err = f"TTS failed: {e}" 
                self.logger.error(err, exc_info=True)
            if os.path.exists(path): 
                try: 
                    os.remove(path) 
                except OSError: 
                    pass
            return {"error": err, "audio_object": None}

    def generate_music(self, description: str, duration: float) -> Dict:
        if not self.pipeline.music_pipeline:
            return {"error": "MusicGen pipeline unavailable", "audio_object": None}
        if duration <= 0:
            return {"error": "Invalid duration", "audio_object": None}

        self.logger.info(f"Generating {duration:.1f}s music (MusicGen) in segments: '{description}'")
        target_sr = TARGET_SAMPLE_RATE
        output_id = f"musicgen_{uuid.uuid4().hex[:8]}"
        output_path = os.path.join(self.pipeline.output_dir, f"{output_id}.wav")
        segment_duration = 30.0  # Generate 30-sec chunks (adjustable)
        overlap_sec = 2.0  # Crossfade duration between segments
        max_tokens_per_segment = int(segment_duration * MUSICGEN_TOKENS_PER_SECOND)

        # Calculate number of segments
        num_segments = int(np.ceil(duration / (segment_duration - overlap_sec)))
        self.logger.debug(f"Breaking into {num_segments} segments of {segment_duration}s each")

        # Generate music segments
        segments = []
        for i in range(num_segments):
            # Adjust prompt slightly for variety (optional)
            seg_prompt = f"{description} (part {i+1})" if num_segments > 1 else description
            self.logger.debug(f"Generating segment {i+1}/{num_segments}: '{seg_prompt}'")
            
            try:
                forward_params = {"max_new_tokens": max_tokens_per_segment}
                music_output = self.pipeline.music_pipeline(seg_prompt, forward_params=forward_params)

                if not music_output or "audio" not in music_output or "sampling_rate" not in music_output:
                    raise ValueError("MusicGen returned invalid data.")

                audio_data = music_output["audio"]
                original_sr = music_output["sampling_rate"]
                if isinstance(audio_data, list):
                    audio_data = audio_data[0]
                if isinstance(audio_data, torch.Tensor):
                    audio_data = audio_data.cpu().numpy()
                audio_array = np.array(audio_data).squeeze()
                if audio_array.ndim > 1 and audio_array.shape[0] > 0 and audio_array.shape[-1] > 1:
                    audio_array = np.mean(audio_array, axis=-1)  # Stereo to mono
                audio_array = audio_array.astype(np.float32)

                # Resample if needed
                if original_sr != target_sr:
                    num_samples = int(len(audio_array) * target_sr / original_sr)
                    original_time = np.linspace(0, len(audio_array) / original_sr, len(audio_array), endpoint=False)
                    resampled_time = np.linspace(0, len(audio_array) / original_sr, num_samples, endpoint=False)
                    audio_array = np.interp(resampled_time, original_time, audio_array).astype(np.float32)

                # Convert to AudioSegment
                audio_seg = AudioSegment(
                    data=audio_array.tobytes(),
                    sample_width=audio_array.dtype.itemsize,
                    frame_rate=target_sr,
                    channels=1
                )
                segments.append(audio_seg)
                self.logger.debug(f"Segment {i+1} generated: {audio_seg.duration_seconds:.2f}s")

            except Exception as e:
                self.logger.error(f"Segment {i+1} failed: {e}", exc_info=True)
                return {"error": f"MusicGen segment {i+1} failed: {e}", "audio_object": None}

        # Combine segments with crossfades
        if not segments:
            return {"error": "No segments generated", "audio_object": None}

        self.logger.info("Combining music segments with crossfades...")
        combined_audio = segments[0]
        for i, next_seg in enumerate(segments[1:], 1):
            # Trim combined_audio if longer than needed after overlap
            overlap_ms = int(overlap_sec * 1000)
            if combined_audio.duration_seconds > duration:
                combined_audio = combined_audio[:int(duration * 1000)]
                break
            # Crossfade with next segment
            combined_audio = combined_audio.append(next_seg, crossfade=overlap_ms)
            self.logger.debug(f"Combined up to segment {i+1}: {combined_audio.duration_seconds:.2f}s")

        # Trim or pad to match desired duration
        final_duration = combined_audio.duration_seconds
        if final_duration < duration:
            self.logger.debug(f"Padding audio from {final_duration:.2f}s to {duration:.2f}s")
            combined_audio += AudioSegment.silent(duration=int((duration - final_duration) * 1000), frame_rate=target_sr)
        elif final_duration > duration:
            self.logger.debug(f"Trimming audio from {final_duration:.2f}s to {duration:.2f}s")
            combined_audio = combined_audio[:int(duration * 1000)]

        # Export final audio
        try:
            combined_audio.export(output_path, format="wav")
            self.logger.info(f"Music generated: {output_path}, Duration: {combined_audio.duration_seconds:.2f}s")
            return {
                "file_path": output_path,
                "duration": combined_audio.duration_seconds,
                "sample_rate": target_sr,
                "audio_object": combined_audio
            }
        except Exception as e:
            self.logger.error(f"Export failed: {e}", exc_info=True)
            return {"error": f"Music export failed: {e}", "audio_object": None}

    def generate_or_select_sfx(self, description: str, target_duration_sec: float) -> Dict:
        if self.pipeline.freesound_api_key:
            result = self._fetch_from_freesound(description, target_duration_sec, False)
            if result and result.get("file_path"): 
                return result
        if GENERATE_PLACEHOLDER_SFX: 
            return self._generate_placeholder_sfx(description, max(1.0, target_duration_sec))
        return {"error": f"SFX '{description}' not found", "audio_object": None}

    def _fetch_from_freesound(self, search_query: str, target_duration_sec: float, is_music: bool) -> dict:
        if not self.pipeline.freesound_api_key: 
            return {"error": "Freesound key missing", "audio_object": None}
        target_sr = TARGET_SAMPLE_RATE
        uuid_hex = uuid.uuid4().hex[:8]
        prefix = "music" if is_music else "sfx"
        f_path = os.path.join(self.pipeline.output_dir, f"{prefix}_{uuid_hex}.wav")
        t_path = os.path.join(self.pipeline.output_dir, f"temp_{prefix}_{uuid_hex}.mp3")
        audio_obj = None
        try:
            min_d = target_duration_sec * 0.5 if not is_music else target_duration_sec
            max_d = target_duration_sec * 1.5 if not is_music else 180
            min_d = max(0.5, min_d)
            max_d = max(min_d + 5, max_d)
            filter = f"duration:[{min_d:.1f} TO {max_d:.1f}]"
            params = {"query":search_query, "filter":filter, "token":self.pipeline.freesound_api_key, "fields":"id,name,previews,license", "sort":"rating_desc" if is_music else "score", "page_size":5}
            resp = requests.get("https://freesound.org/apiv2/search/text/", params = params, headers = {"User-Agent": "AudioPipeline/1.3"}, timeout = FREESOUND_SEARCH_TIMEOUT)
            resp.raise_for_status()
            results = resp.json().get("results",[])
            if not results: 
                return {"error": "No results", "audio_object": None}
            sel = None
            url = None
            for s in results: 
                url = s.get("previews",{}).get("preview-hq-mp3") or s.get("previews",{}).get("preview-lq-mp3") 
                if url: 
                    sel = s
                    break
            if not url: 
                return {"error": "No preview URL", "audio_object": None}
            with requests.get(url, stream = True, timeout = FREESOUND_DOWNLOAD_TIMEOUT) as r: 
                r.raise_for_status()
                open(t_path,"wb").write(r.content)
            audio_obj = AudioSegment.from_file(t_path)
            if audio_obj.channels>1: 
                audio_obj = audio_obj.set_channels(1)
            if audio_obj.frame_rate!=target_sr: 
                audio_obj = audio_obj.set_frame_rate(target_sr)
            audio_obj.export(f_path, format = "wav") 
            duration = audio_obj.duration_seconds
            meta = {"source":"freesound", "id":sel.get("id"), "license":sel.get("license", "?")}
            return {"file_path":f_path, "duration":duration, "sample_rate":target_sr, "metadata":meta, "audio_object":audio_obj}
        except Exception as e: 
            err = f"Freesound failed: {e}"
            self.logger.error(err, exc_info = True)
            return {"error":err, "audio_object":None}
        finally:
             if os.path.exists(t_path): 
                try: 
                    os.remove(t_path) 
                except OSError: 
                    pass

    def _generate_placeholder_sfx(self, description: str, duration_sec: float) -> Dict:
        target_sr = TARGET_SAMPLE_RATE
        path = os.path.join(self.pipeline.output_dir, f"sfx_placeholder_{uuid.uuid4().hex[:6]}.wav")
        audio_obj = None
        try:
            freq = 440
            amp = 0.3
            n_samples = int(duration_sec * target_sr)
            if n_samples <= 0: 
                raise ValueError("Zero samples")
            t = np.linspace(0., duration_sec, n_samples, endpoint = False)
            data = amp * np.sin(2. * np.pi * freq * t)
            fade = min(int(0.05 * target_sr), n_samples // 20)
            if fade > 1: 
                data[:fade] *= np.linspace(0., 1., fade)
                data[-fade:] *= np.linspace(1., 0., fade)
            data = data.astype(np.float32)
            audio_obj = AudioSegment(data = data.tobytes(), sample_width = data.dtype.itemsize, frame_rate = target_sr, channels = 1)
            sf.write(path, data, target_sr)
            dur = len(data) / target_sr
            return {"file_path": path, "duration": dur, "sample_rate": target_sr, "metadata": {"source":"placeholder"}, "audio_object": audio_obj}
        except Exception as e: 
            err = f"Placeholder failed: {e}"
            self.logger.error(err, exc_info = True)
            return {"error":err, "audio_object":None}

class Synchronizer:
    def __init__(self, pipeline: AudioPipeline):
        self.pipeline = pipeline
        self.logger = pipeline.logger

    def build_timeline(self, valid_segments: List[ScriptSegment], target_sr: int) -> Optional[ProductionTimeline]:
        self.logger.info(f"Building timeline from {len(valid_segments)} segments.")
        elements: List[AudioElement] = []
        current_t = TIMELINE_START_PADDING_SEC
        pause = DEFAULT_PAUSE_SEC

        if not valid_segments:
            return ProductionTimeline([], [], 0.0, target_sr)
        for seg in valid_segments:
            audio_obj = getattr(seg, '_temp_audio_object', None)
            if not seg.voice_file or not os.path.exists(seg.voice_file) or not seg.duration or seg.duration <= 0:
                continue
            st = current_t
            et = st + seg.duration
            vol = SPEECH_VOLUME
            f_in = DEFAULT_FADE_SEC
            f_out = DEFAULT_FADE_SEC
            desc = None
            if seg.segment_type == 'sfx':
                vol = SFX_VOLUME
                f_in = SFX_FADE_IN_SEC
                f_out = SFX_FADE_OUT_SEC
                desc = seg.text
            elif seg.segment_type in ['dialogue', 'narration']:
                desc = f"{seg.speaker}: {seg.text[:60]}..."

            elem = AudioElement(
                id = f"{seg.segment_type}_{seg.id}",
                audio_type = seg.segment_type,
                file_path = seg.voice_file,
                start_time = st,
                duration = seg.duration,
                end_time = et,
                volume = vol,
                fade_in = f_in,
                fade_out = f_out,
                description = desc,
                audio_object = audio_obj)
            elements.append(elem)
            seg.start_time = st
            seg.end_time = et
            current_t = et + pause

        max_et = max((e.end_time for e in elements if e.end_time is not None), default = 0.0)
        duration = max_et + TIMELINE_END_PADDING_SEC
        clean_segments = []
        for s in valid_segments:
             if hasattr(s, '_temp_audio_object'):
                  delattr(s, '_temp_audio_object')
             clean_segments.append(s)
        self.logger.info(f"Timeline built: {len(elements)} elements, duration ~{duration:.2f}s")
        return ProductionTimeline(clean_segments, elements, duration, target_sr)

    def adjust_timing(self, timeline: ProductionTimeline) -> ProductionTimeline:
        adj = [e for e in timeline.audio_elements if e.audio_type != 'music']
        music = [e for e in timeline.audio_elements if e.audio_type == 'music']
        if not adj:
            timeline.duration = max((m.end_time for m in music if m.end_time), default = 0.0) + TIMELINE_END_PADDING_SEC
            return timeline
        adj.sort(key = lambda x: x.start_time)
        min_p = MIN_PAUSE_SEC
        adjusted = False
        for i in range(1, len(adj)):
            curr = adj[i]
            prev = adj[i-1]
            if prev.end_time and curr.start_time:
                pause = curr.start_time - prev.end_time
                desired_st = prev.end_time + min_p
                if pause < min_p - 0.001:
                    shift = desired_st - curr.start_time
                    for k in range(i, len(adj)): 
                        adj[k].start_time += shift
                        adj[k].end_time += shift
                    adjusted = True
        timeline.audio_elements = sorted(adj + music, key = lambda x: x.start_time)
        max_non_music = max((e.end_time for e in adj if e.end_time), default = 0.0)
        max_music = max((m.end_time for m in music if m.end_time), default = 0.0)
        timeline.duration = max(max_non_music, max_music) + TIMELINE_END_PADDING_SEC
        self.logger.info(f"{'Timing adjusted.' if adjusted else 'No adjustments.'} Final duration: {timeline.duration:.2f}s")
        return timeline


# --- Audio Compilation Class ---
class AudioCompiler:
    def __init__(self, pipeline: AudioPipeline):
        self.pipeline = pipeline
        self.logger = pipeline.logger

    def compile_timeline(self, timeline: ProductionTimeline, output_path: str) -> Optional[str]:
        target_sr = timeline.sample_rate
        self.logger.info(f"Compiling to: {output_path} (SR: {target_sr}Hz)")
        if not timeline.audio_elements: 
            self.logger.error("No elements.")
            return None
        if timeline.duration <= 0: 
            self.logger.error("Zero duration.")
            return None

        last_content_end = max((e.end_time for e in timeline.audio_elements if e.audio_type not in ['music','direction'] and e.end_time and e.volume > 0.001), default = 0.0)
        music_target_end = last_content_end + MUSIC_FADE_START_OFFSET_SEC + MUSIC_FINAL_FADE_DURATION_SEC
        canvas_dur = max(timeline.duration, music_target_end if last_content_end > 0 else timeline.duration)
        duration_ms = int(canvas_dur * 1000)

        try:
            final_audio = AudioSegment.silent(duration_ms, target_sr)
        except Exception as e:
            self.logger.error(f"Failed canvas creation: {e}")
            return None

        timeline.audio_elements.sort(key = lambda x: x.start_time)
        for elem in timeline.audio_elements:
            if elem.audio_type == 'direction':
                continue
            if not elem.duration or elem.duration <= 0:
                continue
            audio: Optional[AudioSegment] = None
            try:
                if elem.audio_object and isinstance(elem.audio_object, AudioSegment):
                    self.logger.debug(f"Using in-memory object for {elem.id}")
                    audio = elem.audio_object
                elif elem.file_path and os.path.exists(elem.file_path):
                    self.logger.debug(f"Loading {elem.id} from file: {elem.file_path}")
                    ext = os.path.splitext(elem.file_path)[1].lower()
                    hint = ext[1:] if ext in ['.wav','.mp3'] else None
                    try:
                         audio = AudioSegment.from_file(elem.file_path, format=hint)
                    except Exception:
                         audio = AudioSegment.from_file(elem.file_path) # Retry without hint
                else:
                    self.logger.error(f"Missing audio source for {elem.id}. Skipping element.")
                    continue # Skip element if no source
                if audio is None:
                    raise ValueError("Audio object is None after load attempts.")

                # --- Processing ---
                if audio.frame_rate != target_sr: 
                    audio = audio.set_frame_rate(target_sr)
                if audio.channels > 1: 
                    audio = audio.set_channels(1)
                len_s = audio.duration_seconds
                len_ms = len(audio)
                if len_s <= 0: 
                    self.logger.warning(f"Zero duration after load {elem.id}. Skip.")
                    continue

                # Music Trimming/Fading
                if elem.audio_type == "music" and last_content_end > 0:
                    target_len_s = music_target_end
                    if len_s > target_len_s + 0.1: 
                        audio = audio[:int(target_len_s * 1000)]
                    elem.fade_out = MUSIC_FINAL_FADE_DURATION_SEC

                # Volume Adjustment
                if elem.volume < 0.001:
                    audio = AudioSegment.silent(len_ms, target_sr)
                elif abs(elem.volume - 1.0) > 0.01:
                     try:
                         gain_db = pydub_utils.ratio_to_db(float(elem.volume))
                         audio = audio + gain_db
                     except Exception as vol_err:
                          self.logger.error(f"Invalid volume '{elem.volume}' for {elem.id}: {vol_err}.")

                # Fades
                f_in = min(int(elem.fade_in*1000), len_ms // 2) if elem.fade_in * 1000 > 1 else 0
                f_out = min(int(elem.fade_out*1000), len_ms // 2) if elem.fade_out * 1000 > 1 else 0
                if f_in > 1:
                    audio = audio.fade_in(f_in)
                if f_out > 1: 
                    audio = audio.fade_out(f_out)

                # Overlay
                pos_ms = int(elem.start_time * 1000)
                if pos_ms < duration_ms:
                    final_audio = final_audio.overlay(audio, position = pos_ms)
                else:
                    self.logger.warning(f"Skip overlay {elem.id}: pos {pos_ms} > canvas {duration_ms}")

            except Exception as e:
                self.logger.error(f"Failed processing {elem.id}: {e}", exc_info=True)

        # Final Normalization & Export
        if NORMALIZE_OUTPUT and final_audio.duration_seconds > 0:
             try:
                 dbfs = final_audio.dBFS
                 if dbfs > -np.inf:
                     final_audio = final_audio.apply_gain(NORMALIZATION_TARGET_DBFS - dbfs)
             except Exception as e: 
                 self.logger.error(f"Normalization failed: {e}")
        try:
            final_audio.export(output_path, format = OUTPUT_FORMAT)
            return output_path
        except Exception as e:
            self.logger.error(f"Export failed: {e}")
            return None

# --- Main Execution ---
if __name__ == "__main__":
    print("--- Audio Pipeline (MusicGen, Formatted, Cleaned) ---")

    if not os.path.exists(SCRIPT_FILE):
        print(f"ERROR: Input script file '{SCRIPT_FILE}' not found!")
        exit()

    print(f"Using script: '{SCRIPT_FILE}'")
    print(f"Output directory: '{OUTPUT_DIR}'")

    openai_key_set = bool(os.getenv("OPENAI_API_KEY"))
    freesound_key_set = bool(os.getenv("FREESOUND_API_KEY"))
    print(f"\nAPI Keys: OpenAI TTS ({openai_key_set}), Freesound ({freesound_key_set})")
    if not openai_key_set: 
        print("WARNING: OpenAI TTS disabled.")
    if not freesound_key_set: 
        print("WARNING: Freesound SFX disabled.")
    print(f"\nProcessing script...")
    final_file = None
    try:
        pipeline_instance = AudioPipeline()
        if not pipeline_instance.openai_client and not pipeline_instance.music_pipeline and not pipeline_instance.freesound_api_key:
             print("ERROR: No generation capabilities available (OpenAI/MusicGen/Freesound failed or keys missing).")
        else:
             final_file = pipeline_instance.process_script_to_audio(script_path = SCRIPT_FILE)

    except Exception as e:
        print("\n--- CRITICAL ERROR ---")
        logging.exception("Unhandled exception:")
        print(f"Pipeline failed: {e}")

    if final_file:
        print("\n--- SUCCESS ---")
        print(f"Generated: {final_file}")
    else:
        print("\n--- FAILURE ---")
        print("Check logs for details.")
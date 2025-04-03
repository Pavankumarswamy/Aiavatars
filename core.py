import os
import sys
from functools import lru_cache
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Directory setup
now_dir = os.getcwd()
sys.path.append(now_dir)


# Cache VoiceConverter
@lru_cache(maxsize=None)
def import_voice_converter():
    """Import and return VoiceConverter from RVC."""
    try:
        from rvc.infer.infer import VoiceConverter
        return VoiceConverter()
    except ImportError as e:
        logger.error(f"Failed to import VoiceConverter: {e}")
        raise


# Inference function
def run_infer_script(
    pitch: int,
    filter_radius: int,
    index_rate: float,
    volume_envelope: float,  # Renamed from rms_mix_rate for clarity
    protect: float,
    hop_length: int,
    f0_method: str,
    input_path: str,
    output_path: str,
    pth_path: str,
    index_path: str,
    split_audio: bool,
    f0_autotune: bool,
    f0_autotune_strength: float,
    clean_audio: bool,
    clean_strength: float,
    export_format: str,
    f0_file: str = None,
    embedder_model: str = "contentvec",
    embedder_model_custom: str = None,
    formant_shifting: bool = False,
    formant_qfrency: float = 1.0,
    formant_timbre: float = 1.0,
    post_process: bool = False,
    reverb: bool = False,
    pitch_shift: bool = False,
    limiter: bool = False,
    gain: bool = False,
    distortion: bool = False,
    chorus: bool = False,
    bitcrush: bool = False,
    clipping: bool = False,
    compressor: bool = False,
    delay: bool = False,
    reverb_room_size: float = 0.5,
    reverb_damping: float = 0.5,
    reverb_wet_gain: float = 0.5,
    reverb_dry_gain: float = 0.5,
    reverb_width: float = 0.5,
    reverb_freeze_mode: float = 0.5,
    pitch_shift_semitones: float = 0.0,
    limiter_threshold: float = -6.0,
    limiter_release_time: float = 0.01,
    gain_db: float = 0.0,
    distortion_gain: float = 25.0,
    chorus_rate: float = 1.0,
    chorus_depth: float = 0.25,
    chorus_center_delay: float = 7.0,
    chorus_feedback: float = 0.0,
    chorus_mix: float = 0.5,
    bitcrush_bit_depth: int = 8,
    clipping_threshold: float = -6.0,
    compressor_threshold: float = 0.0,
    compressor_ratio: float = 1.0,
    compressor_attack: float = 1.0,
    compressor_release: float = 100.0,
    delay_seconds: float = 0.5,
    delay_feedback: float = 0.0,
    delay_mix: float = 0.5,
    sid: int = 0,
) -> tuple[str, str]:
    """
    Run voice conversion inference on a single audio file.

    Args:
        pitch (int): Pitch adjustment in semitones.
        filter_radius (int): Median filter radius for pitch smoothing.
        index_rate (float): Influence of index file on output (0 to 1).
        volume_envelope (float): Blend level of output volume envelope (0 to 1).
        protect (float): Protection level for consonants (0 to 0.5).
        hop_length (int): Hop length for pitch extraction.
        f0_method (str): Pitch extraction algorithm.
        input_path (str): Path to input audio file.
        output_path (str): Path to save output audio file.
        pth_path (str): Path to RVC model (.pth file).
        index_path (str): Path to index file (.index).
        split_audio (bool): Whether to split audio into segments.
        f0_autotune (bool): Apply autotune to pitch.
        f0_autotune_strength (float): Strength of autotune (0 to 1).
        clean_audio (bool): Apply noise reduction.
        clean_strength (float): Strength of noise reduction (0 to 1).
        export_format (str): Output audio format (e.g., WAV, MP3).
        f0_file (str, optional): Path to external F0 file.
        embedder_model (str): Model for speaker embeddings.
        embedder_model_custom (str, optional): Path to custom embedder model.
        formant_shifting (bool): Apply formant shifting.
        formant_qfrency (float): Formant shifting quefrency.
        formant_timbre (float): Formant shifting timbre.
        post_process (bool): Apply post-processing effects.
        reverb (bool): Apply reverb effect.
        pitch_shift (bool): Apply pitch shift effect.
        limiter (bool): Apply limiter effect.
        gain (bool): Apply gain adjustment.
        distortion (bool): Apply distortion effect.
        chorus (bool): Apply chorus effect.
        bitcrush (bool): Apply bitcrush effect.
        clipping (bool): Apply clipping effect.
        compressor (bool): Apply compressor effect.
        delay (bool): Apply delay effect.
        reverb_room_size (float): Reverb room size (0 to 1).
        reverb_damping (float): Reverb damping (0 to 1).
        reverb_wet_gain (float): Reverb wet gain (0 to 1).
        reverb_dry_gain (float): Reverb dry gain (0 to 1).
        reverb_width (float): Reverb stereo width (0 to 1).
        reverb_freeze_mode (float): Reverb freeze mode (0 to 1).
        pitch_shift_semitones (float): Pitch shift in semitones.
        limiter_threshold (float): Limiter threshold in dB.
        limiter_release_time (float): Limiter release time in seconds.
        gain_db (float): Gain adjustment in dB.
        distortion_gain (float): Distortion gain level.
        chorus_rate (float): Chorus rate in Hz.
        chorus_depth (float): Chorus depth (0 to 1).
        chorus_center_delay (float): Chorus center delay in ms.
        chorus_feedback (float): Chorus feedback (0 to 1).
        chorus_mix (float): Chorus mix level (0 to 1).
        bitcrush_bit_depth (int): Bitcrush bit depth.
        clipping_threshold (float): Clipping threshold in dB.
        compressor_threshold (float): Compressor threshold in dB.
        compressor_ratio (float): Compressor ratio.
        compressor_attack (float): Compressor attack time in ms.
        compressor_release (float): Compressor release time in ms.
        delay_seconds (float): Delay time in seconds.
        delay_feedback (float): Delay feedback (0 to 1).
        delay_mix (float): Delay mix level (0 to 1).
        sid (int): Speaker ID for multi-speaker models.

    Returns:
        tuple[str, str]: (status_message, output_file_path)
    """
    try:
        # Validate input paths
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input audio file not found: {input_path}")
        if not os.path.exists(pth_path):
            raise FileNotFoundError(f"Model file not found: {pth_path}")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")

        # Prepare arguments for VoiceConverter
        kwargs = {
            "audio_input_path": input_path,
            "audio_output_path": output_path,
            "model_path": pth_path,
            "index_path": index_path,
            "pitch": pitch,
            "filter_radius": filter_radius,
            "index_rate": index_rate,
            "volume_envelope": volume_envelope,
            "protect": protect,
            "hop_length": hop_length,
            "f0_method": f0_method,
            "split_audio": split_audio,
            "f0_autotune": f0_autotune,
            "f0_autotune_strength": f0_autotune_strength,
            "clean_audio": clean_audio,
            "clean_strength": clean_strength,
            "export_format": export_format,
            "f0_file": f0_file,
            "embedder_model": embedder_model,
            "embedder_model_custom": embedder_model_custom,
            "formant_shifting": formant_shifting,
            "formant_qfrency": formant_qfrency,
            "formant_timbre": formant_timbre,
            "post_process": post_process,
            "reverb": reverb,
            "pitch_shift": pitch_shift,
            "limiter": limiter,
            "gain": gain,
            "distortion": distortion,
            "chorus": chorus,
            "bitcrush": bitcrush,
            "clipping": clipping,
            "compressor": compressor,
            "delay": delay,
            "reverb_room_size": reverb_room_size,
            "reverb_damping": reverb_damping,
            "reverb_wet_level": reverb_wet_gain,
            "reverb_dry_level": reverb_dry_gain,
            "reverb_width": reverb_width,
            "reverb_freeze_mode": reverb_freeze_mode,
            "pitch_shift_semitones": pitch_shift_semitones,
            "limiter_threshold": limiter_threshold,
            "limiter_release": limiter_release_time,
            "gain_db": gain_db,
            "distortion_gain": distortion_gain,
            "chorus_rate": chorus_rate,
            "chorus_depth": chorus_depth,
            "chorus_delay": chorus_center_delay,
            "chorus_feedback": chorus_feedback,
            "chorus_mix": chorus_mix,
            "bitcrush_bit_depth": bitcrush_bit_depth,
            "clipping_threshold": clipping_threshold,
            "compressor_threshold": compressor_threshold,
            "compressor_ratio": compressor_ratio,
            "compressor_attack": compressor_attack,
            "compressor_release": compressor_release,
            "delay_seconds": delay_seconds,
            "delay_feedback": delay_feedback,
            "delay_mix": delay_mix,
            "sid": sid,
        }

        # Run inference
        infer_pipeline = import_voice_converter()
        infer_pipeline.convert_audio(**kwargs)

        # Adjust output path based on export format
        output_file = output_path if export_format.lower() == "wav" else output_path.replace(".wav", f".{export_format.lower()}")

        logger.info(f"Inference completed for {input_path}")
        return f"File {input_path} inferred successfully.", output_file

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return f"Error during inference: {str(e)}", ""


if __name__ == "__main__":
    # Minimal test case for debugging
    result, output = run_infer_script(
        pitch=0,
        filter_radius=3,
        index_rate=0.5,
        volume_envelope=0.47,
        protect=0.432,
        hop_length=64,
        f0_method="fcpe",
        input_path=os.path.join(now_dir, "assets", "audios", "test.wav"),
        output_path=os.path.join(now_dir, "assets", "audios", "test_output.wav"),
        pth_path=os.path.join(now_dir, "logs", "CBN_AI_FINAL", "CBN_AI_FINAL_447e_1788s_best_epoch.pth"),
        index_path=os.path.join(now_dir, "logs", "CBN_AI_FINAL", "added_IVF123_Flat_nprobe_1.index"),
        split_audio=False,
        f0_autotune=False,
        f0_autotune_strength=1.0,
        clean_audio=False,
        clean_strength=0.67,
        export_format="WAV",
    )
    print(result, output)
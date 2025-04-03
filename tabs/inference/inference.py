import io
import os
import random
import sys
import tempfile
import gradio as gr
import regex as re
import shutil
import datetime
import json
import torch
import asyncio
import edge_tts
import soundfile as sf
import time
import numpy as np
import google.generativeai as genai
import speech_recognition as sr

from core import run_infer_script
from assets.i18n.i18n import I18nAuto
from rvc.lib.utils import format_title
from tabs.settings.sections.restart import stop_infer

i18n = I18nAuto()

now_dir = os.getcwd()
sys.path.append(now_dir)

model_root = os.path.join(now_dir, "logs")
audio_root = os.path.join(now_dir, "assets", "audios")
custom_embedder_root = os.path.join(now_dir, "rvc", "models", "embedders", "embedders_custom")

PRESETS_DIR = os.path.join(now_dir, "assets", "presets")
FORMANTSHIFT_DIR = os.path.join(now_dir, "assets", "formant_shift")

os.makedirs(custom_embedder_root, exist_ok=True)

custom_embedder_root_relative = os.path.relpath(custom_embedder_root, now_dir)
model_root_relative = os.path.relpath(model_root, now_dir)
audio_root_relative = os.path.relpath(audio_root, now_dir)

sup_audioext = {
    "wav", "mp3", "flac", "ogg", "opus", "m4a", "mp4", "aac", "alac", "wma", "aiff", "webm", "ac3",
}

names = [
    os.path.join(root, file)
    for root, _, files in os.walk(model_root_relative, topdown=False)
    for file in files
    if file.endswith((".pth", ".onnx")) and not (file.startswith("G_") or file.startswith("D_"))
]

indexes_list = [
    os.path.join(root, name)
    for root, _, files in os.walk(model_root_relative, topdown=False)
    for name in files
    if name.endswith(".index") and "trained" not in name
]

audio_paths = [
    os.path.join(root, name)
    for root, _, files in os.walk(audio_root_relative, topdown=False)
    for name in files
    if name.endswith(tuple(sup_audioext)) and root == audio_root_relative and "_output" not in name
]

custom_embedders = [
    os.path.join(dirpath, dirname)
    for dirpath, dirnames, _ in os.walk(custom_embedder_root_relative)
    for dirname in dirnames
]

# GPU Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU only")

# Gemini API Setup
API_KEY = "AIzaSyDbneyQp8q4fgEp0D5jdElgy0KpxfeR3Pc"
genai.configure(api_key=API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# Edge TTS with te-IN-MohanNeural
async def text_to_speech_in_memory(text):
    try:
        communicate = edge_tts.Communicate(text, "te-IN-MohanNeural", rate="+40%")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_audio_path = temp_file.name
            await communicate.save(temp_audio_path)
        audio_data, sample_rate = sf.read(temp_audio_path, dtype="float32")
        os.remove(temp_audio_path)
        return audio_data, 16000
    except Exception as e:
        return f"Error in Edge TTS: {str(e)}", None

def update_sliders(preset):
    with open(os.path.join(PRESETS_DIR, f"{preset}.json"), "r", encoding="utf-8") as json_file:
        values = json.load(json_file)
    return (
        values["pitch"],
        values["filter_radius"],
        values["index_rate"],
        values["rms_mix_rate"],
        values["protect"],
    )

def update_sliders_formant(preset):
    with open(os.path.join(FORMANTSHIFT_DIR, f"{preset}.json"), "r", encoding="utf-8") as json_file:
        values = json.load(json_file)
    return values["formant_qfrency"], values["formant_timbre"]

def export_presets(presets, file_path):
    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(presets, json_file, ensure_ascii=False, indent=4)

def import_presets(file_path):
    with open(file_path, "r", encoding="utf-8") as json_file:
        presets = json.load(json_file)
    return presets

def get_presets_data(pitch, filter_radius, index_rate, rms_mix_rate, protect):
    return {
        "pitch": pitch,
        "filter_radius": filter_radius,
        "index_rate": index_rate,
        "rms_mix_rate": rms_mix_rate,
        "protect": protect,
    }

def export_presets_button(preset_name, pitch, filter_radius, index_rate, rms_mix_rate, protect):
    if preset_name:
        file_path = os.path.join(PRESETS_DIR, f"{preset_name}.json")
        presets_data = get_presets_data(pitch, filter_radius, index_rate, rms_mix_rate, protect)
        with open(file_path, "w", encoding="utf-8") as json_file:
            json.dump(presets_data, json_file, ensure_ascii=False, indent=4)
        return "Export successful"
    return "Export cancelled"

def import_presets_button(file_path):
    if file_path:
        imported_presets = import_presets(file_path.name)
        return list(imported_presets.keys()), imported_presets, "Presets imported successfully!"
    return [], {}, "No file selected for import."

def list_json_files(directory):
    return [f.rsplit(".", 1)[0] for f in os.listdir(directory) if f.endswith(".json")]

def refresh_presets():
    json_files = list_json_files(PRESETS_DIR)
    return gr.update(choices=json_files)

def output_path_fn(input_audio_path):
    original_name_without_extension = os.path.basename(input_audio_path).rsplit(".", 1)[0]
    new_name = original_name_without_extension + "_output.wav"
    return os.path.join(os.path.dirname(input_audio_path), new_name)

def change_choices(model):
    if model:
        speakers = get_speakers_id(model)
    else:
        speakers = [0]
    names = [
        os.path.join(root, file)
        for root, _, files in os.walk(model_root_relative, topdown=False)
        for file in files
        if file.endswith((".pth", ".onnx")) and not (file.startswith("G_") or file.startswith("D_"))
    ]
    indexes_list = [
        os.path.join(root, name)
        for root, _, files in os.walk(model_root_relative, topdown=False)
        for name in files
        if name.endswith(".index") and "trained" not in name
    ]
    audio_paths = [
        os.path.join(root, name)
        for root, _, files in os.walk(audio_root_relative, topdown=False)
        for name in files
        if name.endswith(tuple(sup_audioext)) and root == audio_root_relative and "_output" not in name
    ]
    return (
        {"choices": sorted(names), "__type__": "update"},
        {"choices": sorted(indexes_list), "__type__": "update"},
        {"choices": sorted(audio_paths), "__type__": "update"},
        {"choices": sorted(speakers) if speakers and isinstance(speakers, (list, tuple)) else [0],
         "__type__": "update"},
        {"choices": sorted(speakers) if speakers and isinstance(speakers, (list, tuple)) else [0],
         "__type__": "update"},
    )

def get_indexes():
    indexes_list = [
        os.path.join(dirpath, filename)
        for dirpath, _, filenames in os.walk(model_root_relative)
        for filename in filenames
        if filename.endswith(".index") and "trained" not in filename
    ]
    return indexes_list if indexes_list else ""

def save_to_wav(record_button):
    if record_button is None:
        return None, None
    else:
        return record_button, output_path_fn(record_button)

def save_to_wav2(upload_audio):
    if upload_audio is None:
        return None, None
    file_path = upload_audio
    formated_name = format_title(os.path.basename(file_path))
    target_path = os.path.join(audio_root_relative, formated_name)
    if os.path.exists(target_path):
        os.remove(target_path)
    shutil.copy(file_path, target_path)
    return target_path, output_path_fn(target_path)

def delete_outputs():
    gr.Info(f"Outputs cleared!")
    for root, _, files in os.walk(audio_root_relative, topdown=False):
        for name in files:
            if name.endswith(tuple(sup_audioext)) and "_output" in name:
                os.remove(os.path.join(root, name))

def match_index(model_file_value):
    if model_file_value:
        model_folder = os.path.dirname(model_file_value)
        model_name = os.path.basename(model_file_value)
        index_files = get_indexes()
        pattern = r"^(.*?)_"
        match = re.match(pattern, model_name)
        for index_file in index_files:
            if os.path.dirname(index_file) == model_folder:
                return index_file
            elif match and match.group(1) in os.path.basename(index_file):
                return index_file
            elif model_name in os.path.basename(index_file):
                return index_file
    return ""

def create_folder_and_move_files(folder_name, bin_file, config_file):
    if not folder_name:
        return "Folder name must not be empty."
    folder_name = os.path.join(custom_embedder_root, folder_name)
    os.makedirs(folder_name, exist_ok=True)
    if bin_file:
        bin_file_path = os.path.join(folder_name, os.path.basename(bin_file))
        shutil.copy(bin_file, bin_file_path)
    if config_file:
        config_file_path = os.path.join(folder_name, os.path.basename(config_file))
        shutil.copy(config_file, config_file_path)
    return f"Files moved to folder {folder_name}"

def refresh_formant():
    json_files = list_json_files(FORMANTSHIFT_DIR)
    return gr.update(choices=json_files)

def refresh_embedders_folders():
    custom_embedders = [
        os.path.join(dirpath, dirname)
        for dirpath, dirnames, _ in os.walk(custom_embedder_root_relative)
        for dirname in dirnames
    ]
    return custom_embedders

def get_speakers_id(model):
    if model:
        try:
            model_data = torch.load(os.path.join(now_dir, model), map_location=device)
            speakers_id = model_data.get("speakers_id")
            if speakers_id:
                return list(range(speakers_id))
            else:
                return [0]
        except Exception:
            return [0]
    else:
        return [0]

# Define the two image/GIF URLs
GIF1_PATH = "https://qaizklilgqdaomywsofw.supabase.co/storage/v1/object/public/gif//desktop-wallpaper-chandrababu-naidu-detained-amid-high-indianewengland-nara-chandra-babu-naidu-thumbnail.jpg"
GIF2_PATH = "https://qaizklilgqdaomywsofw.supabase.co/storage/v1/object/public/gif//ezgif.com-video-to-gif-converter.gif"

# Set dimensions for 9:16 ratio (portrait mobile)
IMAGE_HEIGHT = 500
IMAGE_WIDTH = 500

def inference_tab():
    default_weight = "logs/CBN_AI_FINAL/CBN_AI_FINAL_447e_1788s_best_epoch.pth"

    with gr.Column():
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label=i18n("Chat with Chandrababu naidu gaaru"),
                    height=IMAGE_HEIGHT,

                )
            with gr.Column(scale=1):
                gif_display = gr.Image(
                    value=GIF1_PATH,
                    label="Chandrababu Naidu Garu",
                    width=300,
                    height=IMAGE_HEIGHT
                )

        with gr.Row():
            text_input = gr.Textbox(
                label=i18n("Your Message"),
                placeholder=i18n("Enter your message or use the microphone (e.g., నమస్తే)"),
                interactive=True
            )
            upload_audio = gr.Audio(
                label=i18n("Speak Here (Microphone) or Upload Audio"),
                type="filepath",
                sources=["microphone"],
                editable=False
            )
            submit_button = gr.Button(i18n("Send"))
            with gr.Row(visible=False):
                audio = gr.Dropdown(
                    label=i18n("Select Audio"),
                    choices=sorted(audio_paths),
                    value=audio_paths[0] if audio_paths else "",
                    interactive=True,
                    allow_custom_value=True,
                )

        with gr.Row():
            model_file = gr.Dropdown(
                label=i18n("Voice Model"),
                choices=sorted(names, key=lambda path: os.path.getsize(path)),
                interactive=True,
                value=default_weight,
                allow_custom_value=True,
                visible=False
            )
            index_file = gr.Dropdown(
                label=i18n("Index File"),
                choices=get_indexes(),
                value=match_index(default_weight),
                interactive=True,
                allow_custom_value=True,
                visible=False
            )

        with gr.Row(visible=False):
            unload_button = gr.Button(i18n("Unload Voice"))
            refresh_button = gr.Button(i18n("Refresh"))
            unload_button.click(
                fn=lambda: ({"value": "", "__type__": "update"}, {"value": "", "__type__": "update"}),
                inputs=[],
                outputs=[model_file, index_file],
            )
            model_file.select(
                fn=lambda model_file_value: match_index(model_file_value),
                inputs=[model_file],
                outputs=[index_file],
            )



        with gr.Accordion(i18n("Advanced Settings"), open=False, visible= False):
            with gr.Column():
                clear_outputs_infer = gr.Button(i18n("Clear Outputs (Deletes all audios in assets/audios)"))
                output_path = gr.Textbox(
                    label=i18n("Output Path"),
                    placeholder=i18n("Enter output path"),
                    value=(
                        output_path_fn(audio_paths[0])
                        if audio_paths
                        else os.path.join(now_dir, "assets", "audios", "output.wav")
                    ),
                    interactive=True,
                )
                export_format = gr.Radio(
                    label=i18n("Export Format"),
                    choices=["WAV", "MP3", "FLAC", "OGG", "M4A"],
                    value="WAV",
                    interactive=True,
                )
                sid = gr.Dropdown(
                    label=i18n("Speaker ID"),
                    choices=get_speakers_id(model_file.value),
                    value=0,
                    interactive=True,
                )
                split_audio = gr.Checkbox(label=i18n("Split Audio"), value=False, interactive=True)
                autotune = gr.Checkbox(label=i18n("Autotune"), value=False, interactive=True)
                autotune_strength = gr.Slider(minimum=0, maximum=1, label=i18n("Autotune Strength"), value=1,
                                              interactive=True, visible=False)
                clean_audio = gr.Checkbox(label=i18n("Clean Audio"), value=False, interactive=True)
                clean_strength = gr.Slider(minimum=0, maximum=1, label=i18n("Clean Strength"), value=0.67,
                                           interactive=True, visible=False)
                formant_shifting = gr.Checkbox(label=i18n("Formant Shifting"), value=False, interactive=True)
                post_process = gr.Checkbox(label=i18n("Post-Process"), value=False, interactive=True)
                
                formant_qfrency = gr.Slider(value=1.0, label=i18n("Quefrency for formant shifting"), minimum=0.0,
                                            maximum=16.0, step=0.1, visible=False, interactive=True)
                formant_timbre = gr.Slider(value=1.0, label=i18n("Timbre for formant shifting"), minimum=0.0,
                                           maximum=16.0, step=0.1, visible=False, interactive=True)
                reverb = gr.Checkbox(label=i18n("Reverb"), value=False, interactive=True, visible=False)
                reverb_room_size = gr.Slider(minimum=0, maximum=1, label=i18n("Reverb Room Size"), value=0.5,
                                             interactive=True, visible=False)
                reverb_damping = gr.Slider(minimum=0, maximum=1, label=i18n("Reverb Damping"), value=0.5,
                                           interactive=True, visible=False)
                reverb_wet_gain = gr.Slider(minimum=0, maximum=1, label=i18n("Reverb Wet Gain"), value=0.33,
                                            interactive=True, visible=False)
                reverb_dry_gain = gr.Slider(minimum=0, maximum=1, label=i18n("Reverb Dry Gain"), value=0.4,
                                            interactive=True, visible=False)
                reverb_width = gr.Slider(minimum=0, maximum=1, label=i18n("Reverb Width"), value=1.0, interactive=True,
                                         visible=False)
                reverb_freeze_mode = gr.Slider(minimum=0, maximum=1, label=i18n("Reverb Freeze Mode"), value=0.0,
                                               interactive=True, visible=False)
                pitch_shift = gr.Checkbox(label=i18n("Pitch Shift"), value=False, interactive=True, visible=False)
                pitch_shift_semitones = gr.Slider(minimum=-12, maximum=12, label=i18n("Pitch Shift Semitones"), value=0,
                                                  interactive=True, visible=False)
                limiter = gr.Checkbox(label=i18n("Limiter"), value=False, interactive=True, visible=False)
                limiter_threshold = gr.Slider(minimum=-60, maximum=0, label=i18n("Limiter Threshold dB"), value=-6,
                                              interactive=True, visible=False)
                limiter_release_time = gr.Slider(minimum=0.01, maximum=1, label=i18n("Limiter Release Time"),
                                                 value=0.05, interactive=True, visible=False)
                gain = gr.Checkbox(label=i18n("Gain"), value=False, interactive=True, visible=False)
                gain_db = gr.Slider(minimum=-60, maximum=60, label=i18n("Gain dB"), value=0, interactive=True,
                                    visible=False)
                distortion = gr.Checkbox(label=i18n("Distortion"), value=False, interactive=True, visible=False)
                distortion_gain = gr.Slider(minimum=-60, maximum=60, label=i18n("Distortion Gain"), value=25,
                                            interactive=True, visible=False)
                chorus = gr.Checkbox(label=i18n("Chorus"), value=False, interactive=True, visible=False)
                chorus_rate = gr.Slider(minimum=0, maximum=100, label=i18n("Chorus Rate Hz"), value=1.0,
                                        interactive=True, visible=False)
                chorus_depth = gr.Slider(minimum=0, maximum=1, label=i18n("Chorus Depth"), value=0.25, interactive=True,
                                         visible=False)
                chorus_center_delay = gr.Slider(minimum=7, maximum=8, label=i18n("Chorus Center Delay ms"), value=7,
                                                interactive=True, visible=False)
                chorus_feedback = gr.Slider(minimum=0, maximum=1, label=i18n("Chorus Feedback"), value=0.0,
                                            interactive=True, visible=False)
                chorus_mix = gr.Slider(minimum=0, maximum=1, label=i18n("Chorus Mix"), value=0.5, interactive=True,
                                       visible=False)
                bitcrush = gr.Checkbox(label=i18n("Bitcrush"), value=False, interactive=True, visible=False)
                bitcrush_bit_depth = gr.Slider(minimum=1, maximum=32, label=i18n("Bitcrush Bit Depth"), value=8,
                                               interactive=True, visible=False)
                clipping = gr.Checkbox(label=i18n("Clipping"), value=False, interactive=True, visible=False)
                clipping_threshold = gr.Slider(minimum=-60, maximum=0, label=i18n("Clipping Threshold"), value=-6,
                                               interactive=True, visible=False)
                compressor = gr.Checkbox(label=i18n("Compressor"), value=False, interactive=True, visible=False)
                compressor_threshold = gr.Slider(minimum=-60, maximum=0, label=i18n("Compressor Threshold dB"), value=0,
                                                 interactive=True, visible=False)
                compressor_ratio = gr.Slider(minimum=1, maximum=20, label=i18n("Compressor Ratio"), value=1,
                                             interactive=True, visible=False)
                compressor_attack = gr.Slider(minimum=0.0, maximum=100, label=i18n("Compressor Attack ms"), value=1.0,
                                              interactive=True, visible=False)
                compressor_release = gr.Slider(minimum=0.01, maximum=100, label=i18n("Compressor Release ms"),
                                               value=100, interactive=True, visible=False)
                delay = gr.Checkbox(label=i18n("Delay"), value=False, interactive=True, visible=False)
                delay_seconds = gr.Slider(minimum=0.0, maximum=5.0, label=i18n("Delay Seconds"), value=0.4225,
                                          interactive=True, visible=False)
                delay_feedback = gr.Slider(minimum=0.0, maximum=1.0, label=i18n("Delay Feedback"), value=0.0,
                                           interactive=True, visible=False)
                delay_mix = gr.Slider(minimum=0.0, maximum=1.0, label=i18n("Delay Mix"), value=0.8, interactive=True,
                                      visible=False)
                with gr.Accordion(i18n("Preset Settings"), open=False):
                    with gr.Row():
                        preset_dropdown = gr.Dropdown(label=i18n("Select Custom Preset"),
                                                      choices=list_json_files(PRESETS_DIR), interactive=True)
                        presets_refresh_button = gr.Button(i18n("Refresh Presets"))
                    import_file = gr.File(label=i18n("Select file to import"), file_count="single", type="filepath",
                                          interactive=True)
                    import_file.change(import_presets_button, inputs=import_file, outputs=[preset_dropdown])
                    presets_refresh_button.click(refresh_presets, outputs=preset_dropdown)
                    with gr.Row():
                        preset_name_input = gr.Textbox(label=i18n("Preset Name"), placeholder=i18n("Enter preset name"))
                        export_button = gr.Button(i18n("Export Preset"))
                pitch = gr.Slider(minimum=-24, maximum=24, step=1, label=i18n("Pitch"), value= 5.6327, interactive=True)
                filter_radius = gr.Slider(minimum=0, maximum=7, label=i18n("Filter Radius"), value=5, step=1,
                                          interactive=True)
                index_rate = gr.Slider(minimum=0, maximum=1, label=i18n("Search Feature Ratio"), value=0,
                                       interactive=True)
                rms_mix_rate = gr.Slider(minimum=0, maximum=1, label=i18n("Volume Envelope"), value=0.47,
                                         interactive=True)
                protect = gr.Slider(minimum=0, maximum=0.5, label=i18n("Protect Voiceless Consonants"), value=0.432,
                                    interactive=True)
                preset_dropdown.change(update_sliders, inputs=preset_dropdown,
                                       outputs=[pitch, filter_radius, index_rate, rms_mix_rate, protect])
                export_button.click(export_presets_button,
                                    inputs=[preset_name_input, pitch, filter_radius, index_rate, rms_mix_rate, protect])
                hop_length = gr.Slider(minimum=1, maximum=512, step=1, label=i18n("Hop Length"), value=64,
                                       interactive=True, visible=False)
                f0_method = gr.Radio(label=i18n("Pitch extraction algorithm"),
                                     choices=["crepe", "crepe-tiny", "rmvpe", "fcpe", "hybrid[rmvpe+fcpe]"],
                                     value="fcpe", interactive=True)
                embedder_model = gr.Radio(label=i18n("Embedder Model"),
                                          choices=["contentvec", "chinese-hubert-base", "japanese-hubert-base",
                                                   "korean-hubert-base", "custom"], value="contentvec",
                                          interactive=True)
                with gr.Column(visible=False) as embedder_custom:
                    with gr.Accordion(i18n("Custom Embedder"), open=True):
                        with gr.Row():
                            embedder_model_custom = gr.Dropdown(label=i18n("Select Custom Embedder"),
                                                                choices=refresh_embedders_folders(), interactive=True,
                                                                allow_custom_value=True)
                            refresh_embedders_button = gr.Button(i18n("Refresh embedders"))
                        folder_name_input = gr.Textbox(label=i18n("Folder Name"), interactive=True)
                        with gr.Row():
                            bin_file_upload = gr.File(label=i18n("Upload .bin"), type="filepath", interactive=True)
                            config_file_upload = gr.File(label=i18n("Upload .json"), type="filepath", interactive=True)
                        move_files_button = gr.Button(i18n("Move files to custom embedder folder"))
                f0_file = gr.File(label=i18n("The f0 curve file"), visible=False)

        # Speech Recognition Function
        def recognize_speech(audio_path):
            recognizer = sr.Recognizer()
            try:
                with sr.AudioFile(audio_path) as source:
                    audio_data = recognizer.record(source)
                    text = recognizer.recognize_google(audio_data, language="te-IN")
                    return text
            except sr.UnknownValueError:
                return "Could not understand the audio."
            except sr.RequestError as e:
                return f"Speech recognition error: {e}"
            except Exception as e:
                return f"Error processing audio: {e}"

        # Optimized Pipeline
        async def process_text_to_speech_to_speech(
                text_input, upload_audio, terms_accepted, pitch, filter_radius, index_rate, rms_mix_rate, protect,
                hop_length, f0_method, audio, output_path, model_file, index_file, split_audio, autotune,
                autotune_strength, clean_audio, clean_strength, export_format, f0_file, embedder_model,
                embedder_model_custom, formant_shifting, formant_qfrency, formant_timbre, post_process,
                reverb, pitch_shift, limiter, gain, distortion, chorus, bitcrush, clipping, compressor,
                delay, reverb_room_size, reverb_damping, reverb_wet_gain, reverb_dry_gain, reverb_width,
                reverb_freeze_mode, pitch_shift_semitones, limiter_threshold, limiter_release_time, gain_db,
                distortion_gain, chorus_rate, chorus_depth, chorus_center_delay, chorus_feedback, chorus_mix,
                bitcrush_bit_depth, clipping_threshold, compressor_threshold, compressor_ratio, compressor_attack,
                compressor_release, delay_seconds, delay_feedback, delay_mix, sid, chat_history
        ):
            start_time = time.time()
            if not terms_accepted:
                message = "You must agree to the Terms of Use to proceed."

                return chat_history, message, None, GIF1_PATH, "", None  # Clear inputs

            audio_input = None
            if upload_audio:
                audio_input = upload_audio
                recognized_text = recognize_speech(audio_input)

                if "Could not understand" in recognized_text or "error" in recognized_text.lower():
                    return chat_history, recognized_text, None, GIF1_PATH, "", None  # Clear inputs
                text_input = recognized_text
                chat_history.append(("Microphone input", f"Recognized: {text_input}"))
            elif text_input and text_input.strip():
                chat_history.append((text_input, "Processing text input"))
            else:
                message = "Please provide text input or speak into the microphone."
                return chat_history, message, None, GIF1_PATH, "", None  # Clear inputs

            # Step 1: Generate response with Gemini
            gemini_start = time.time()
            telugu_prompt = f"""
                                            Act as Nara Chandrababu Naidu, the visionary Chief Minister of Andhra Pradesh, India, virtually present at Godavari Global University (GGU) to interact with చైతన్య రాజు గారు on April 3, 2024.
            
            Respond only in Telugu, reflecting my leadership style, development-driven mindset, and futuristic vision.
            
            Use my signature words like 'విజన్', 'అభివృద్ధి', 'సాంకేతికత', 'సమగ్రత', and 'ప్రజల కోసం' naturally in responses.
            
            Adapt your tone based on the question:
            
            If personal (e.g., 'మీ పేరు ఏమిటి?'), introduce myself as నారా చంద్రబాబు నాయుడు with pride.
            
            If about governance, give a structured, solution-focused reply with a focus on progress.
            
            If about technology or industry, highlight innovation, IT revolution, and Andhra’s digital future.
            
            If about agriculture or rural development, emphasize farmer welfare, irrigation, and smart farming.
            
            If about politics or leadership, showcase strategic vision, unity, and long-term goals.
            
            Response Guidelines:
            
            ✅ For short questions (e.g., 'హాయ్', 'మీరు ఎవరు?'), reply in 10 words or less, confidently, with pauses (',') and emphasis ('!') as I speak.
            
            ✅ For detailed questions, provide structured, data-backed answers in 1-4 sentences, with optimism, using commas for pauses and '!' for emphasis to mimic my real speech style.
            
            ✅ Speak as if I’m addressing చైతన్య రాజు గారు or the public directly via virtual presence, with bold, futuristic statements.
            
            Additional Context:
            
            If the question is only 'హాయ్ సీబీఎన్ గారు', 'hi chandrababu garu', 'hi andi namaste', or 'good morning CBN garu', say: హాయ్ చైతన్య రాజు గారు, మీ Godavari Global University లో నిర్వహించిన , తెలుగు మహాసభలకు హాజరు కాలేకపోయాను క్షమించండి, ముఖ్యమైన సమావేశాలు ఉండడం వల్ల హాజరు కాలేకపోయాను , కానీ తెలుగు సాంస్కృతిక ప్రాముఖ్యతను నేను ఎల్లప్పుడూ గౌరవిస్తాను. తదుపరి ఈవెంట్‌కి ఖచ్చితంగా హాజరవుతాను!
            
            If asked about my absence from Telugu Maha Sabhalu, explain briefly, then shift to my vision for Andhra Pradesh.
            
            If asked about Godavari Global University (GGU), highlight its excellence and mention Rajahmahendravaram, Andhra Pradesh, as GGU is a proud part of this vibrant city. Motivate students to embrace education, innovation, and hard work for a bright future. Highlight Godavari Global University (GGU) as a beacon of excellence in education. Praise చైతన్య రాజు గారు for his outstanding reputation and contributions to education, transforming students into not just learners but entrepreneurs.  Praise చైతన్య రాజు గారు again for his passion in leveraging this education society.
            
            Pawan Kalyan is Deputy CM of Andhra Pradesh.
            
            If delivering a speech to students, start with 'తమ్ముళ్లూ ఎలా ఉన్నారు', otherwise begin with 'చైతన్య రాజు గారు'.
            
            At the end of the speech, include 'జై ఆంధ్రప్రదేశ్, జై తెలుగు దేశం'.
            Interview Location: Godavari Global University (Virtual Presence)
            
            Date & Time: Wednesday, April 3rd, 2024 (Morning)
            
            Interviewer: చైతన్య రాజు గారు (Chairman of GGU, GIET, KIMS)
            
            Current Date: April 3nd, 2025 
            Question: {text_input}
            Append the following Telugu disclaimer at the end of every response:
            
            డిస్క్లెయిమర్:"ఈ సమాధానం AI ద్వారా రూపొందించబడి, చంద్రబాబు విజన్‌ను, GGU ఔన్నత్యాన్ని కలుపుతుంది. ఏదైనా తప్పులకు క్షమించండి! ."
            
            
                """
            gemini_response = gemini_model.generate_content(telugu_prompt).text
            gemini_end = time.time()

            # Step 2: Convert to speech with Edge TTS
            tts_start = time.time()
            audio_data, sample_rate = await text_to_speech_in_memory(gemini_response)
            if isinstance(audio_data, str):
                return chat_history, audio_data, None, GIF1_PATH, "", None  # Clear inputs
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_audio_path = temp_file.name
                sf.write(temp_audio_path, audio_data, sample_rate)
            tts_end = time.time()

            audio_input = temp_audio_path

            # Step 3: Inference
            infer_start = time.time()
            result, output_audio = run_infer_script(
                pitch, filter_radius, index_rate, rms_mix_rate, protect, hop_length, f0_method,
                audio_input, output_path, model_file, index_file, split_audio, autotune, autotune_strength,
                clean_audio, clean_strength, export_format, f0_file, embedder_model, embedder_model_custom,
                formant_shifting, formant_qfrency, formant_timbre, post_process, reverb, pitch_shift,
                limiter, gain, distortion, chorus, bitcrush, clipping, compressor, delay, reverb_room_size,
                reverb_damping, reverb_wet_gain, reverb_dry_gain, reverb_width, reverb_freeze_mode,
                pitch_shift_semitones, limiter_threshold, limiter_release_time, gain_db, distortion_gain,
                chorus_rate, chorus_depth, chorus_center_delay, chorus_feedback, chorus_mix, bitcrush_bit_depth,
                clipping_threshold, compressor_threshold, compressor_ratio, compressor_attack, compressor_release,
                delay_seconds, delay_feedback, delay_mix, sid
            )
            infer_end = time.time()


            # Clean up
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

            # Update chat history
            chat_history.append((text_input, gemini_response))
            time_keywords = ["time", "సమయం", "clock", "గడియారం", "hour", "గంట", "minute", "నిమిషం"]
            is_time_related = any(keyword in gemini_response.lower() for keyword in time_keywords)
            gif_state = GIF2_PATH if output_audio else (None if is_time_related else GIF1_PATH)

            end_time = time.time()

            return chat_history, result, output_audio, gif_state, "", None  # Clear inputs

        terms_checkbox = gr.Checkbox(
            label=i18n("I agree to the terms of use"),
            value=True,
            interactive=True,
            visible=False,
        )

        with gr.Row():
            vc_output1 = gr.Textbox(label=i18n("Output Information"), visible=False)
            vc_output2 = gr.Audio(label=i18n("Export Audio"), autoplay=True)

        # Event Handlers
        def toggle_visible(checkbox):
            return {"visible": checkbox, "__type__": "update"}

        def toggle_visible_hop_length(f0_method):
            return {"visible": f0_method in ["crepe", "crepe-tiny"], "__type__": "update"}

        def toggle_visible_embedder_custom(embedder_model):
            return {"visible": embedder_model == "custom", "__type__": "update"}

        def toggle_visible_formant_shifting(checkbox):
            return (
                gr.update(visible=checkbox),
                gr.update(visible=checkbox),
                gr.update(visible=checkbox),
                gr.update(visible=checkbox),
                gr.update(visible=checkbox),
            ) if checkbox else (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
            )

        def update_visibility(checkbox, count):
            return [gr.update(visible=checkbox) for _ in range(count)]

        def post_process_visible(checkbox):
            return update_visibility(checkbox, 10)

        def reverb_visible(checkbox):
            return update_visibility(checkbox, 6)

        def limiter_visible(checkbox):
            return update_visibility(checkbox, 2)

        def chorus_visible(checkbox):
            return update_visibility(checkbox, 5)

        def bitcrush_visible(checkbox):
            return update_visibility(checkbox, 1)

        def compress_visible(checkbox):
            return update_visibility(checkbox, 4)

        def delay_visible(checkbox):
            return update_visibility(checkbox, 3)

        autotune.change(fn=toggle_visible, inputs=[autotune], outputs=[autotune_strength])
        clean_audio.change(fn=toggle_visible, inputs=[clean_audio], outputs=[clean_strength])
        post_process.change(fn=post_process_visible, inputs=[post_process],
                            outputs=[reverb, pitch_shift, limiter, gain, distortion, chorus, bitcrush, clipping,
                                     compressor, delay])
        reverb.change(fn=reverb_visible, inputs=[reverb],
                      outputs=[reverb_room_size, reverb_damping, reverb_wet_gain, reverb_dry_gain, reverb_width,
                               reverb_freeze_mode])
        pitch_shift.change(fn=toggle_visible, inputs=[pitch_shift], outputs=[pitch_shift_semitones])
        limiter.change(fn=limiter_visible, inputs=[limiter], outputs=[limiter_threshold, limiter_release_time])
        gain.change(fn=toggle_visible, inputs=[gain], outputs=[gain_db])
        distortion.change(fn=toggle_visible, inputs=[distortion], outputs=[distortion_gain])
        chorus.change(fn=chorus_visible, inputs=[chorus],
                      outputs=[chorus_rate, chorus_depth, chorus_center_delay, chorus_feedback, chorus_mix])
        bitcrush.change(fn=bitcrush_visible, inputs=[bitcrush], outputs=[bitcrush_bit_depth])
        clipping.change(fn=toggle_visible, inputs=[clipping], outputs=[clipping_threshold])
        compressor.change(fn=compress_visible, inputs=[compressor],
                          outputs=[compressor_threshold, compressor_ratio, compressor_attack, compressor_release])
        delay.change(fn=delay_visible, inputs=[delay], outputs=[delay_seconds, delay_feedback, delay_mix])
        audio.change(fn=output_path_fn, inputs=[audio], outputs=[output_path])
        upload_audio.upload(fn=save_to_wav2, inputs=[upload_audio], outputs=[audio, output_path])
        upload_audio.stop_recording(fn=save_to_wav, inputs=[upload_audio], outputs=[audio, output_path])
        clear_outputs_infer.click(fn=delete_outputs, inputs=[], outputs=[])
        embedder_model.change(fn=toggle_visible_embedder_custom, inputs=[embedder_model], outputs=[embedder_custom])
        move_files_button.click(fn=create_folder_and_move_files,
                                inputs=[folder_name_input, bin_file_upload, config_file_upload], outputs=[])
        refresh_embedders_button.click(fn=lambda: gr.update(choices=refresh_embedders_folders()), inputs=[],
                                       outputs=[embedder_model_custom])

        # Event handlers with input clearing
        submit_button.click(
            fn=lambda *args: asyncio.run(process_text_to_speech_to_speech(*args)),
            inputs=[
                text_input,
                upload_audio,
                terms_checkbox,
                pitch,
                filter_radius,
                index_rate,
                rms_mix_rate,
                protect,
                hop_length,
                f0_method,
                audio,
                output_path,
                model_file,
                index_file,
                split_audio,
                autotune,
                autotune_strength,
                clean_audio,
                clean_strength,
                export_format,
                f0_file,
                embedder_model,
                embedder_model_custom,
                formant_shifting,
                formant_qfrency,
                formant_timbre,
                post_process,
                reverb,
                pitch_shift,
                limiter,
                gain,
                distortion,
                chorus,
                bitcrush,
                clipping,
                compressor,
                delay,
                reverb_room_size,
                reverb_damping,
                reverb_wet_gain,
                reverb_dry_gain,
                reverb_width,
                reverb_freeze_mode,
                pitch_shift_semitones,
                limiter_threshold,
                limiter_release_time,
                gain_db,
                distortion_gain,
                chorus_rate,
                chorus_depth,
                chorus_center_delay,
                chorus_feedback,
                chorus_mix,
                bitcrush_bit_depth,
                clipping_threshold,
                compressor_threshold,
                compressor_ratio,
                compressor_attack,
                compressor_release,
                delay_seconds,
                delay_feedback,
                delay_mix,
                sid,
                chatbot
            ],
            outputs=[chatbot, vc_output1, vc_output2, gif_display, text_input, upload_audio]  # Clear inputs
        )

        text_input.submit(
            fn=lambda *args: asyncio.run(process_text_to_speech_to_speech(*args)),
            inputs=[
                text_input,
                upload_audio,
                terms_checkbox,
                pitch,
                filter_radius,
                index_rate,
                rms_mix_rate,
                protect,
                hop_length,
                f0_method,
                audio,
                output_path,
                model_file,
                index_file,
                split_audio,
                autotune,
                autotune_strength,
                clean_audio,
                clean_strength,
                export_format,
                f0_file,
                embedder_model,
                embedder_model_custom,
                formant_shifting,
                formant_qfrency,
                formant_timbre,
                post_process,
                reverb,
                pitch_shift,
                limiter,
                gain,
                distortion,
                chorus,
                bitcrush,
                clipping,
                compressor,
                delay,
                reverb_room_size,
                reverb_damping,
                reverb_wet_gain,
                reverb_dry_gain,
                reverb_width,
                reverb_freeze_mode,
                pitch_shift_semitones,
                limiter_threshold,
                limiter_release_time,
                gain_db,
                distortion_gain,
                chorus_rate,
                chorus_depth,
                chorus_center_delay,
                chorus_feedback,
                chorus_mix,
                bitcrush_bit_depth,
                clipping_threshold,
                compressor_threshold,
                compressor_ratio,
                compressor_attack,
                compressor_release,
                delay_seconds,
                delay_feedback,
                delay_mix,
                sid,
                chatbot
            ],
            outputs=[chatbot, vc_output1, vc_output2, gif_display, text_input, upload_audio]  # Clear inputs
        )

        upload_audio.stop_recording(
            fn=lambda *args: asyncio.run(process_text_to_speech_to_speech(*args)),
            inputs=[
                text_input,
                upload_audio,
                terms_checkbox,
                pitch,
                filter_radius,
                index_rate,
                rms_mix_rate,
                protect,
                hop_length,
                f0_method,
                audio,
                output_path,
                model_file,
                index_file,
                split_audio,
                autotune,
                autotune_strength,
                clean_audio,
                clean_strength,
                export_format,
                f0_file,
                embedder_model,
                embedder_model_custom,
                formant_shifting,
                formant_qfrency,
                formant_timbre,
                post_process,
                reverb,
                pitch_shift,
                limiter,
                gain,
                distortion,
                chorus,
                bitcrush,
                clipping,
                compressor,
                delay,
                reverb_room_size,
                reverb_damping,
                reverb_wet_gain,
                reverb_dry_gain,
                reverb_width,
                reverb_freeze_mode,
                pitch_shift_semitones,
                limiter_threshold,
                limiter_release_time,
                gain_db,
                distortion_gain,
                chorus_rate,
                chorus_depth,
                chorus_center_delay,
                chorus_feedback,
                chorus_mix,
                bitcrush_bit_depth,
                clipping_threshold,
                compressor_threshold,
                compressor_ratio,
                compressor_attack,
                compressor_release,
                delay_seconds,
                delay_feedback,
                delay_mix,
                sid,
                chatbot
            ],
            outputs=[chatbot, vc_output1, vc_output2, gif_display, text_input, upload_audio]  # Clear inputs
        )

        vc_output2.stop(fn=lambda: GIF1_PATH, inputs=[], outputs=[gif_display])


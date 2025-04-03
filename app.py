# import gradio as gr
# import sys
# import os
# import logging
#
# # Constants
# DEFAULT_PORT = 6969
# MAX_PORT_ATTEMPTS = 10
# APP_TITLE = "Text to Speech Synthesizer by Pavan"
#
# # Setup logging
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# logger = logging.getLogger(__name__)
#
# # Add current directory to sys.path
# sys.path.append(os.getcwd())
#
# # Import inference tabs
# try:
#     from tabs.inference.inference import inference_tab
#     from tabs.inference.inference1 import inference_tab1
#     from tabs.inference.inference2 import inference_tab2
# except ImportError as e:
#     logger.error(f"Failed to import inference tabs: {e}")
#     raise
#
# # Define Gradio interface with Tabs
# with gr.Blocks(title=APP_TITLE, css="footer {display: none !important}") as TTS:
#     with gr.Tabs("CBN GARU"):
#         inference_tab()
#     with gr.Tab("Modi JI"):
#         inference_tab1()
#     with gr.Tab("PAWANKALYAN"):
#         inference_tab2()
#
# def launch_gradio(port: int, inbrowser: bool) -> None:
#     """Launch the Gradio interface with specified options."""
#     try:
#         TTS.launch(
#             share=True,
#             inbrowser=inbrowser,
#             server_port=port,
#             server_name="0.0.0.0",
#             show_error=True,
#             prevent_thread_lock=True,
#             debug=True,
#         )
#         logger.info(f"Gradio launched on port {port}. Check console for public URL.")
#     except Exception as e:
#         logger.error(f"Failed to launch Gradio: {e}")
#         raise
#
# def get_port_from_args() -> int:
#     """Extract port number from command-line arguments."""
#     if "--port" in sys.argv:
#         try:
#             port_index = sys.argv.index("--port") + 1
#             if port_index < len(sys.argv):
#                 return int(sys.argv[port_index])
#         except (ValueError, IndexError):
#             logger.warning(f"Invalid port argument. Using default port {DEFAULT_PORT}")
#     return DEFAULT_PORT
#
# if __name__ == "__main__":
#     inbrowser = "--open" in sys.argv
#     port = get_port_from_args()
#
#     for attempt in range(MAX_PORT_ATTEMPTS):
#         try:
#             launch_gradio(port, inbrowser)
#             break
#         except OSError:
#             port -= 1
#             logger.warning(f"Port conflict. Trying new port {port} ({attempt + 1}/{MAX_PORT_ATTEMPTS})")
#         except Exception as e:
#             logger.error(f"Unexpected error: {e}")
#             break
#     else:
#         logger.error("Failed to launch after multiple attempts. Exiting.")
#         sys.exit(1)
import gradio as gr
import sys
import os

from tabs.inference.inference2 import inference_tab2

# Constants
DEFAULT_PORT = 6969
MAX_PORT_ATTEMPTS = 10

# Add current directory to sys.path
now_dir = os.getcwd()
sys.path.append(now_dir)
# Import Tabs
from tabs.inference.inference import inference_tab



# Define Gradio interface
with gr.Blocks(title="Text to Speech Synthesizer by Pavan", css="footer{display:none !important}") as TTS:
    gr.Markdown("# Text to Speech Synthesizer by Pavan")
    with gr.Tab("CBN GARU"):
        inference_tab()
    with gr.Tab("MODI GARU"):
        inference_tab2()



def launch_gradio(port):
    TTS.launch(
        share= False,
        inbrowser="--open" in sys.argv,
        server_port=port,
    )


def get_port_from_args():
    if "--port" in sys.argv:
        port_index = sys.argv.index("--port") + 1
        if port_index < len(sys.argv):
            return int(sys.argv[port_index])
    return DEFAULT_PORT


if __name__ == "__main__":
    port = get_port_from_args()
    for _ in range(MAX_PORT_ATTEMPTS):
        try:
            launch_gradio(port)
            break
        except OSError:
            print(
                f"Failed to launch on port {port}, trying again on port {port - 1}..."
            )
            port -= 1
        except Exception as error:
            print(f"An error occurred launching Gradio: {error}")
            break

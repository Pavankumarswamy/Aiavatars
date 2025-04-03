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

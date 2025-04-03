# import gradio as gr
# import sys
# import os

# # Add current directory to sys.path (useful for local dev and Spaces)
# now_dir = os.getcwd()
# sys.path.append(now_dir)

# # Import Tabs (ensure these files exist in tabs/inference/)
# from tabs.inference.inference import inference_tab
# from tabs.inference.inference2 import inference_tab2

# # Define Gradio interface
# with gr.Blocks(title="Text to Speech Synthesizer by Pavan", css="footer {display: none !important}") as TTS:
#     gr.Markdown("# Text to Speech Synthesizer by Pavan")
#     with gr.Tab("CBN GARU"):
#         inference_tab()
#     with gr.Tab("MODI GARU"):
#         inference_tab2()

# # Launch function for Hugging Face Spaces
# def launch_gradio():
#     TTS.launch()  # No port or server_name needed; Spaces handles it

# if __name__ == "__main__":
#     try:
#         launch_gradio()
#     except Exception as error:
#         print(f"Error launching Gradio: {error}")
#         raise  # Re-raise to ensure errors are visible in Spaces logs



import os

file_path = "/home/user/app/input.txt"  # Change this to a valid file

# If the path is a directory, create a new file
if os.path.isdir(file_path):
    print(f"'{file_path}' is a directory! Creating a new file instead...")
    file_path = os.path.join(file_path, "newfile.txt")  # Create a proper file path

# Ensure the file exists before reading
if not os.path.exists(file_path):
    with open(file_path, "w") as f:
        f.write("This is a test file.")  # Creating a new file with some content

# Now, the script can safely open the file
with open(file_path, "r") as f:
    content = f.read()
    print("File content:", content)
import base64
import re
from mimetypes import guess_type
import torch
import subprocess
import signal
import os

from inference.constants import ANNOTATION_SUB_FOLDER, IMAGE_COLUMN, ORIGINAL_COLUMN


# Function to encode a local image into data URL 
def encode_image(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"


def replace_punctuation_and_spaces(text):
    # Define a regular expression pattern that matches any punctuation or space
    pattern = r'[^\w]'
    
    # Replace matched characters with underscores
    replaced_text = re.sub(pattern, '_', text)
    
    return replaced_text


def get_gpu_count():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=count', '--format=csv,noheader,nounits'], 
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE, 
                                text=True)
        if result.returncode != 0:
            print(f"Error checking GPU count: {result.stderr}")
            return 0
        
        gpu_count = int(result.stdout.split("\n")[0].strip())
        return gpu_count
    except Exception as e:
        print(f"Exception occurred: {e}")
        return 0
    
def use_gpus():
    gpu_count = get_gpu_count()

    dtype = torch.float

    # Initialize tensors on different devices
    devices = [torch.device(f"cuda:{i}") for i in range(gpu_count)]
    tensors = []

    # Create tensors on each device
    for device in devices:
        tensors.append(
            (torch.rand((300000000), device=device, dtype=dtype), 
            torch.rand((300000000), device=device, dtype=dtype))
        )

    # Main computation loop
    while True:
        for i, (a, b) in enumerate(tensors):
            result = torch.dot(a, b)


if __name__ == "__main__":
    use_gpus()


def launch_nohup_script(script_path="../fast_prompt_alignment/inference/utils.py"):
    """
    Launches a Python script using nohup in the background.

    Parameters:
    script_path (str): The path to the Python script to execute.

    Returns:
    int: The process ID of the launched script.
    """
    command = ["nohup", "python3", script_path]

    # Using subprocess.Popen to execute the command
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE, close_fds=True)

    # Detach the process from the parent
    process.communicate()

    print(f"Process launched with PID: {process.pid}")
    return process.pid


def launch_nohup_script_many_times(script_path="../fast_prompt_alignment/inference/utils.py", num_times=3):
    """
    Launches a Python script using nohup in the background.

    Parameters:
    script_path (str): The path to the Python script to execute.
    num_times (int): The number of times to launch the script.

    Returns:
    list: A list of process IDs of the launched scripts.
    """
    process_ids = []
    if num_times > 0:
        for _ in range(num_times):
            process_ids.append(launch_nohup_script(script_path=script_path))
    return process_ids


def kill_process(pid):
    try:
        os.kill(pid, signal.SIGTERM)  # or signal.SIGKILL
        print(f"Process {pid} has been terminated.")
    except ProcessLookupError:
        print(f"Process {pid} does not exist.")
    except PermissionError:
        print(f"Permission denied to kill process {pid}.")
    except Exception as e:
        print(f"An error occurred: {e}")


def kill_multiple_processes(process_ids):
    for pid in process_ids:
        kill_process(pid)


def save_annotations_as_html(df, output_data_path):
    # Start the HTML string
    html_content = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Text and Images</title>
        <style>
            body { font-family: Arial, sans-serif; }
            .container { display: flex; align-items: center; margin-bottom: 20px; }
            .container img { width: 256px; height: 256px; margin-right: 20px; }
            .separator { border-bottom: 1px solid #ccc; margin: 20px 0; }
            .index { font-weight: bold; margin-right: 10px; }
        </style>
    </head>
    <body>
    <h1>Text and Images</h1>
    '''

    # Iterate over the rows in the DataFrame
    for idx, row in df.iterrows():
        text = row[ORIGINAL_COLUMN]
        image_path = row[IMAGE_COLUMN]
        
        # Add the HTML for this row, including the row index and a separator
        html_content += f'''
        <div class="container">
            <span class="index">{idx + 1}.</span>
            <img src="{image_path}" alt="Image">
            <p>{text}</p>
        </div>
        <div class="separator"></div>
        '''

    # End the HTML string
    html_content += '''
    </body>
    </html>
    '''

    # Write the HTML content to a file
    with open(output_data_path + ANNOTATION_SUB_FOLDER + '.html', 'w') as file:
        file.write(html_content)

    print("HTML file has been created successfully.")
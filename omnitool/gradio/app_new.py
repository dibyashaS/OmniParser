"""
The app contains:
- a new UI for the OmniParser AI Agent.
- 
python app_new.py --windows_host_url localhost:8006 --omniparser_server_url localhost:8000
"""

import os
import io
import shutil
import mimetypes
from datetime import datetime
from enum import StrEnum
from functools import partial
from pathlib import Path
from typing import cast, List, Optional
import argparse
import gradio as gr
from anthropic import APIResponse
from anthropic.types import TextBlock
from anthropic.types.beta import BetaMessage, BetaTextBlock, BetaToolUseBlock
from anthropic.types.tool_use_block import ToolUseBlock
from loop import (
    APIProvider,
    sampling_loop_sync,
)
from tools import ToolResult
import requests
from requests.exceptions import RequestException
import base64
import random
import openai

CONFIG_DIR = Path("~/.anthropic").expanduser()
API_KEY_FILE = CONFIG_DIR / "api_key"

INTRO_TEXT = '''
<div style="text-align: center; margin-bottom: 10px;">
    <h2>OmniParser AI Agent</h2>
    <p>Turn any vision-language model into an AI agent. We currently support <b>OpenAI (4o/o1/o3-mini), DeepSeek (R1), Qwen (2.5VL) or Anthropic Computer Use (Sonnet)</b>.</p>
    <p>Type a message and press send to start OmniTool. Press stop to pause, and press the trash icon in the chat to clear the message history.</p>
    <p>You can also upload files for analysis using the file upload section.</p>
</div>
'''

def parse_arguments():
    parser = argparse.ArgumentParser(description="Gradio App")
    parser.add_argument("--windows_host_url", type=str, default='localhost:8006')
    parser.add_argument("--omniparser_server_url", type=str, default="localhost:8000")
    parser.add_argument("--run_folder", type=str, default="./tmp/outputs")
    return parser.parse_args()
args = parse_arguments()

# Update upload folder from args if provided
RUN_FOLDER = Path(os.path.join(args.run_folder, datetime.now().strftime('%Y%m%d_%H%M')))
RUN_FOLDER.mkdir(parents=True, exist_ok=True)

class Sender(StrEnum):
    USER = "user"
    BOT = "assistant"
    TOOL = "tool"


def load_existing_files():
    """Load all existing files from the uploads folder"""
    files = []
    if RUN_FOLDER.exists():
        for file_path in RUN_FOLDER.iterdir():
            if file_path.is_file():
                files.append(str(file_path))
    return files

def setup_state(state):
    if "messages" not in state:
        state["messages"] = []
    if "model" not in state:
        state["model"] = "omniparser + gpt-4o-orchestrated"
    if "provider" not in state:
        state["provider"] = "openai"
    if "openai_api_key" not in state:  # Fetch API keys from environment variables
        state["openai_api_key"] = os.getenv("OPENAI_API_KEY", "")
    if "anthropic_api_key" not in state:
        state["anthropic_api_key"] = os.getenv("ANTHROPIC_API_KEY", "")
    if "api_key" not in state:
        state["api_key"] = ""
    if "auth_validated" not in state:
        state["auth_validated"] = False
    if "responses" not in state:
        state["responses"] = {}
    if "tools" not in state:
        state["tools"] = {}
    if "only_n_most_recent_images" not in state:
        state["only_n_most_recent_images"] = 2
    if 'chatbot_messages' not in state:
        state['chatbot_messages'] = []
    if 'stop' not in state:
        state['stop'] = False
    if 'uploaded_files' not in state:
        state['uploaded_files'] = []  # Start with an empty list instead of loading existing files

async def main(state):
    """Render loop for Gradio"""
    setup_state(state)
    return "Setup completed"

def validate_auth(provider: APIProvider, api_key: str | None):
    if provider == APIProvider.ANTHROPIC:
        if not api_key:
            return "Enter your Anthropic API key to continue."
    if provider == APIProvider.BEDROCK:
        import boto3

        if not boto3.Session().get_credentials():
            return "You must have AWS credentials set up to use the Bedrock API."
    if provider == APIProvider.VERTEX:
        import google.auth
        from google.auth.exceptions import DefaultCredentialsError

        if not os.environ.get("CLOUD_ML_REGION"):
            return "Set the CLOUD_ML_REGION environment variable to use the Vertex API."
        try:
            google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        except DefaultCredentialsError:
            return "Your google cloud credentials are not set up correctly."

def load_from_storage(filename: str) -> str | None:
    """Load data from a file in the storage directory."""
    try:
        file_path = CONFIG_DIR / filename
        if file_path.exists():
            data = file_path.read_text().strip()
            if data:
                return data
    except Exception as e:
        print(f"Debug: Error loading {filename}: {e}")
    return None

def save_to_storage(filename: str, data: str) -> None:
    """Save data to a file in the storage directory."""
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        file_path = CONFIG_DIR / filename
        file_path.write_text(data)
        # Ensure only user can read/write the file
        file_path.chmod(0o600)
    except Exception as e:
        print(f"Debug: Error saving {filename}: {e}")

def _api_response_callback(response: APIResponse[BetaMessage], response_state: dict):
    response_id = datetime.now().isoformat()
    response_state[response_id] = response

def _tool_output_callback(tool_output: ToolResult, tool_id: str, tool_state: dict):
    tool_state[tool_id] = tool_output

def chatbot_output_callback(message, chatbot_state, hide_images=False, sender="bot"):
    def _render_message(message: str | BetaTextBlock | BetaToolUseBlock | ToolResult, hide_images=False):
    
        print(f"_render_message: {str(message)[:100]}")
        
        if isinstance(message, str):
            return message
        
        is_tool_result = not isinstance(message, str) and (
            isinstance(message, ToolResult)
            or message.__class__.__name__ == "ToolResult"
        )
        if not message or (
            is_tool_result
            and hide_images
            and not hasattr(message, "error")
            and not hasattr(message, "output")
        ):  # return None if hide_images is True
            return
        # render tool result
        if is_tool_result:
            message = cast(ToolResult, message)
            if message.output:
                return message.output
            if message.error:
                return f"Error: {message.error}"
            if message.base64_image and not hide_images:
                # somehow can't display via gr.Image
                # image_data = base64.b64decode(message.base64_image)
                # return gr.Image(value=Image.open(io.BytesIO(image_data)))
                return f'<img src="data:image/png;base64,{message.base64_image}">'

        elif isinstance(message, BetaTextBlock) or isinstance(message, TextBlock):
            # Format reasoning text in a collapsible dropdown
            return f"Next step Reasoning: {message.text}"
            # reasoning_text = message.text
            # return f'''
            # <details>
            #     <summary><Current Step Reasoning (click to expand):</summary>
            #     <div style="padding: 10px; background-color: #f8f9fa; border-radius: 5px; margin-top: 5px;">
            #         <pre>{reasoning_text}</pre>
            #     </div>
            # </details>
            # '''
        elif isinstance(message, BetaToolUseBlock) or isinstance(message, ToolUseBlock):
            # return f"Next I will perform the following action: {message.input}"
            return None
        else:  
            return message

    def _truncate_string(s, max_length=500):
        """Truncate long strings for concise printing."""
        if isinstance(s, str) and len(s) > max_length:
            return s[:max_length] + "..."
        return s
    # processing Anthropic messages
    message = _render_message(message, hide_images)
    
    if sender == "bot":
        chatbot_state.append((None, message))
    else:
        chatbot_state.append((message, None))
    
    # Create a concise version of the chatbot state for printing
    concise_state = [(_truncate_string(user_msg), _truncate_string(bot_msg))
                        for user_msg, bot_msg in chatbot_state]
    # print(f"chatbot_output_callback chatbot_state: {concise_state} (truncated)")

def valid_params(user_input, state):
    """Validate all requirements and return a list of error messages."""
    errors = []
    
    for server_name, url in [('Windows Host', 'localhost:5000'), ('OmniParser Server', args.omniparser_server_url)]:
        try:
            url = f'http://{url}/probe'
            response = requests.get(url, timeout=3)
            if response.status_code != 200:
                errors.append(f"{server_name} is not responding")
        except RequestException as e:
            errors.append(f"{server_name} is not responding")
    
    if not state["api_key"].strip():
        errors.append("LLM API Key is not set")

    if not user_input:
        errors.append("no computer use request provided")
    
    return errors

def process_input(user_input, state):
    # Reset the stop flag
    if state["stop"]:
        state["stop"] = False

    errors = valid_params(user_input, state)
    if errors:
        raise gr.Error("Validation errors: " + ", ".join(errors))
    
    # Append the user message to state["messages"]
    state["messages"].append(
        {
            "role": Sender.USER,
            "content": [TextBlock(type="text", text=user_input)],
        }
    )

    # Append the user's message to chatbot_messages with None for the assistant's reply
    state['chatbot_messages'].append((user_input, None))
    yield state['chatbot_messages'], gr.update()  # Yield to update the chatbot UI with the user's message

    print("state")
    print(state)

    # Run sampling_loop_sync with the chatbot_output_callback
    for loop_msg in sampling_loop_sync(
        model=state["model"],
        provider=state["provider"],
        messages=state["messages"],
        output_callback=partial(chatbot_output_callback, chatbot_state=state['chatbot_messages'], hide_images=False),
        tool_output_callback=partial(_tool_output_callback, tool_state=state["tools"]),
        api_response_callback=partial(_api_response_callback, response_state=state["responses"]),
        api_key=state["api_key"],
        only_n_most_recent_images=state["only_n_most_recent_images"],
        max_tokens=16384,
        omniparser_url=args.omniparser_server_url,
        save_folder=str(RUN_FOLDER)
    ):  
        if loop_msg is None or state.get("stop"):
            # Detect and add new files to the state
            file_choices_update = detect_new_files(state)
            yield state['chatbot_messages'], file_choices_update
            print("End of task. Close the loop.")
            break
            
        yield state['chatbot_messages'], gr.update()  # Yield the updated chatbot_messages to update the chatbot UI
    
    # Final detection of new files
    file_choices_update = detect_new_files(state)
    yield state['chatbot_messages'], file_choices_update

    # --- After model outputs, run evaluation ---
    # Example: collect all model outputs and evaluate
    # This is a placeholder for where you would call the evaluator after model responses
    # model_outputs = [{"model": state["model"], "output": <model_output>}]
    # evaluator_model = state.get("evaluator", "gpt-4o-eval")
    # evaluation_results = evaluate_model_outputs(model_outputs, evaluator_model)
    # print("Evaluation Results:", evaluation_results)
    # You can display these results in the UI as needed

def stop_app(state):
    state["stop"] = True
    return "App stopped"

def get_header_image_base64():
    try:
        # Get the absolute path to the image relative to this script
        script_dir = Path(__file__).parent
        image_path = script_dir.parent.parent / "imgs" / "header_bar_thin.png"
        
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
            return f'data:image/png;base64,{encoded_string}'
    except Exception as e:
        print(f"Failed to load header image: {e}")
        return None

def get_file_viewer_html(file_path=None):
    """Generate HTML to view a file based on its type"""
    if not file_path:
        # Return the VNC viewer iframe
        return f'<iframe src="http://{args.windows_host_url}/vnc.html?view_only=1&autoconnect=1&resize=scale" width="100%" height="580" allow="fullscreen"></iframe>'
    
    file_path = Path(file_path)
    if not file_path.exists():
        return f'<div class="error-message">File not found: {file_path.name}</div>'
    
    # Determine the file type
    mime_type, _ = mimetypes.guess_type(file_path)
    file_type = mime_type.split('/')[0] if mime_type else 'unknown'
    file_extension = file_path.suffix.lower()
    
    # Handle different file types
    if file_type == 'image':
        # For images, display them directly
        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
            return f'<div class="file-viewer"><h3>{file_path.name}</h3><img src="data:{mime_type};base64,{encoded_string}" style="max-width:100%; max-height:500px;"></div>'
    
    elif file_extension in ['.txt', '.py', '.js', '.html', '.css', '.json', '.md', '.csv'] or file_type == 'text':
        # For text files, display the content with syntax highlighting for code
        try:
            content = file_path.read_text(errors='replace')  # Use 'replace' to handle encoding issues
            # Escape HTML characters
            content = content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            
            # Add syntax highlighting class based on file extension
            highlight_class = ""
            if file_extension == '.py':
                highlight_class = "language-python"
            elif file_extension == '.js':
                highlight_class = "language-javascript"
            elif file_extension == '.html':
                highlight_class = "language-html"
            elif file_extension == '.css':
                highlight_class = "language-css"
            elif file_extension == '.json':
                highlight_class = "language-json"
            
            return f'''
            <div class="file-viewer">
                <h3>{file_path.name}</h3>
                <pre class="{highlight_class}" style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow: auto; max-height: 500px; white-space: pre-wrap;"><code>{content}</code></pre>
                <script>
                    // Add basic syntax highlighting with CSS
                    if (document.querySelector('.language-python')) {{
                        const keywords = ['def', 'class', 'import', 'from', 'return', 'if', 'else', 'elif', 'for', 'while', 'try', 'except', 'with', 'as', 'in', 'not', 'and', 'or', 'True', 'False', 'None'];
                        const code = document.querySelector('.language-python code');
                        let html = code.innerHTML;
                        keywords.forEach(keyword => {{
                            const regex = new RegExp('\\\\b' + keyword + '\\\\b', 'g');
                            html = html.replace(regex, `<span style="color: #0000FF; font-weight: bold;">$&</span>`);
                        }});
                        // Highlight strings
                        html = html.replace(/(["'])(?:(?=(\\\\?))\2.)*?\1/g, '<span style="color: #008000;">$&</span>');
                        // Highlight comments
                        html = html.replace(/(#.*)$/gm, '<span style="color: #808080;">$1</span>');
                        code.innerHTML = html;
                    }}
                </script>
            </div>
            '''
        except UnicodeDecodeError:
            return f'<div class="error-message">Cannot display binary file: {file_path.name}</div>'
    
    elif file_type == 'video':
        # For videos, use video tag
        with open(file_path, "rb") as video_file:
            encoded_string = base64.b64encode(video_file.read()).decode()
            return f'''
            <div class="file-viewer">
                <h3>{file_path.name}</h3>
                <video controls style="max-width:100%; max-height:500px;">
                    <source src="data:{mime_type};base64,{encoded_string}" type="{mime_type}">
                    Your browser does not support the video tag.
                </video>
            </div>
            '''
    
    elif file_type == 'audio':
        # For audio, use audio tag
        with open(file_path, "rb") as audio_file:
            encoded_string = base64.b64encode(audio_file.read()).decode()
            return f'''
            <div class="file-viewer">
                <h3>{file_path.name}</h3>
                <audio controls>
                    <source src="data:{mime_type};base64,{encoded_string}" type="{mime_type}">
                    Your browser does not support the audio tag.
                </audio>
            </div>
            '''
    
    elif file_extension == '.pdf':
        # For PDFs, embed them using an iframe with base64 data
        try:
            with open(file_path, "rb") as pdf_file:
                encoded_string = base64.b64encode(pdf_file.read()).decode()
                return f'''
                <div class="file-viewer">
                    <h3>{file_path.name}</h3>
                    <iframe src="data:application/pdf;base64,{encoded_string}" width="100%" height="500px" style="border: none;"></iframe>
                </div>
                '''
        except Exception as e:
            return f'<div class="error-message">Error displaying PDF: {str(e)}</div>'
    
    else:
        # For other file types, show info but can't display
        size_kb = file_path.stat().st_size / 1024
        return f'<div class="file-viewer"><h3>{file_path.name}</h3><p>File type: {mime_type or "Unknown"}</p><p>Size: {size_kb:.2f} KB</p><p>This file type cannot be displayed in the browser.</p></div>'

def handle_file_upload(files, state):
    """Handle file uploads and store them in the upload directory"""
    if not files:
        return gr.update(choices=[])
    
    file_choices = []
    
    for file in files:
        # Get the file name and create a path in the upload directory
        file_name = Path(file.name).name
        file_path = RUN_FOLDER / file_name
        
        # Save the file
        shutil.copy(file.name, file_path)
        
        # Add to the list of uploaded files
        file_path_str = str(file_path)
        file_choices.append((file_name, file_path_str))
        
        # Add to state
        if file_path_str not in state['uploaded_files']:
            state['uploaded_files'].append(file_path_str)
    
    # Update the view file dropdown with all uploaded files
    all_file_choices = [(str(Path(path).name), str(path)) for path in state['uploaded_files']]
    
    return gr.update(choices=all_file_choices)

def toggle_view(view_mode, file_path=None, state=None):
    """Toggle between OmniTool Computer view and file viewer"""
    # If switching to File Viewer mode, detect and add new files to the state
    file_choices_update = gr.update(choices=[])
    if view_mode == "File Viewer" and state is not None:
        file_choices_update = detect_new_files(state)
    
    # Return the appropriate view
    if view_mode == "OmniTool Computer":
        return get_file_viewer_html(), file_choices_update  # This returns the VNC iframe
    else:  # File Viewer mode
        if file_path:
            return get_file_viewer_html(file_path), file_choices_update
        else:
            return get_file_viewer_html(), file_choices_update  # Default to VNC if no file selected

def detect_new_files(state):
    """Detect new files in the uploads folder and add them to the state"""
    new_files_count = 0
    if RUN_FOLDER.exists():
        current_files = set(state['uploaded_files'])
        for file_path in RUN_FOLDER.iterdir():
            if file_path.is_file():
                file_path_str = str(file_path)
                if file_path_str not in current_files:
                    # This is a new file not yet in the state
                    state['uploaded_files'].append(file_path_str)
                    new_files_count += 1
                    print(f"Added new file to state: {file_path_str}")
    
    # Return updated file choices
    file_choices = [(Path(path).name, path) for path in state['uploaded_files']]
    print(f"Detected {new_files_count} new files. Total files in state: {len(state['uploaded_files'])}")
    return gr.update(choices=file_choices)

def refresh_files(state):
    """Refresh the list of files from the current session and detect new files"""
    return detect_new_files(state)

def auto_refresh_files(state):
    """Automatically refresh the list of files from the current session and detect new files"""
    return detect_new_files(state)

def update_only_n_images(new_value, state):
    state["only_n_most_recent_images"] = new_value
    return

def update_provider(new_provider, state):
    state["provider"] = new_provider
    # Optionally, update the API key field if needed
    return state.get("api_key", "")

def update_api_key(new_api_key, state):
    state["api_key"] = new_api_key
    return

def clear_chat(state):
    state["chatbot_messages"] = []
    return []

def view_file(file_path, view_toggle):
    # This function returns the HTML for the file viewer
    return get_file_viewer_html(file_path)

def render_model_cards(evals):
    # Render a card deck for model outputs and evaluations
    html = '<div class="card-deck">'
    for eval in evals:
        html += f'''
        <div class="model-card">
            <h4>{eval['model']}</h4>
            <div class="output"><b>Output:</b> {eval['output']}</div>
            <div class="evaluation"><b>Evaluation:</b> {eval['evaluation']}</div>
            <div class="score">Score: {eval['score']}</div>
        </div>
        '''
    html += '</div>'
    return html

with gr.Blocks(theme=gr.themes.Default()) as demo:
    gr.HTML("""
        <style>
        .no-padding {
            padding: 0 !important;
        }
        .no-padding > div {
            padding: 0 !important;
        }
        .markdown-text p {
            font-size: 18px;
        }
        /* Card deck styles */
        .card-deck {
            display: flex;
            flex-wrap: wrap;
            gap: 24px;
            justify-content: center;
            margin-top: 24px;
        }
        .model-card {
            background: linear-gradient(135deg, #e3f0ff 0%, #f8fbff 100%);
            border-radius: 18px;
            box-shadow: 0 4px 16px rgba(30, 64, 175, 0.10), 0 1.5px 4px rgba(30, 64, 175, 0.08);
            padding: 24px 20px 18px 20px;
            min-width: 320px;
            max-width: 370px;
            flex: 1 1 320px;
            transition: box-shadow 0.2s;
            border: 1.5px solid #b6d0f7;
        }
        .model-card:hover {
            box-shadow: 0 8px 32px rgba(30, 64, 175, 0.18), 0 2px 8px rgba(30, 64, 175, 0.12);
        }
        .model-card h4 {
            margin-top: 0;
            margin-bottom: 8px;
            color: #1e40af;
            font-weight: 700;
            font-size: 1.15rem;
        }
        .model-card .output {
            font-size: 1.05rem;
            margin-bottom: 10px;
            color: #222b3a;
        }
        .model-card .evaluation {
            font-size: 0.98rem;
            color: #3b4a6b;
            margin-bottom: 6px;
        }
        .model-card .score {
            font-size: 1.1rem;
            font-weight: 600;
            color: #2563eb;
            background: #e0eaff;
            border-radius: 8px;
            padding: 2px 10px;
            display: inline-block;
        }
        </style>
    """)
    state = gr.State({})
    
    setup_state(state.value)
    
    header_image = get_header_image_base64()
    if header_image:
        gr.HTML(f'<img src="{header_image}" alt="OmniTool Header" width="100%">', elem_classes="no-padding")
        gr.HTML('<h1 style="text-align: center; font-weight: normal; margin-bottom: 20px;">Omni<span style="font-weight: bold;">Tool</span></h1>')
    else:
        gr.Markdown("# OmniTool", elem_classes="text-center")

    if not os.getenv("HIDE_WARNING", False):
        gr.HTML(INTRO_TEXT, elem_classes="markdown-text")

    with gr.Accordion("Settings", open=True, elem_classes="accordion-header"): 
        with gr.Row():
            with gr.Column():
                # Multi-model selection (CheckboxGroup)
                model_options = [
                    "omniparser + gpt-4o", "omniparser + o1", "omniparser + o3-mini", "omniparser + R1", "omniparser + qwen2.5vl", "claude-3-5-sonnet-20241022", "omniparser + gpt-4o-orchestrated", "omniparser + o1-orchestrated", "omniparser + o3-mini-orchestrated", "omniparser + R1-orchestrated", "omniparser + qwen2.5vl-orchestrated"
                ]
                selected_models = gr.CheckboxGroup(
                    model_options,
                    label="Select Models",
                    value=[model_options[0]],
                    interactive=True
                )
            with gr.Column():
                evaluator = gr.Dropdown(
                    label="Evaluator Model",
                    choices=["gpt-4o-eval", "claude-3-opus-eval", "llama-3-70b-eval"],
                    value="gpt-4o-eval",
                    interactive=True
                )
            with gr.Column():
                only_n_images = gr.Slider(
                    label="N most recent screenshots",
                    minimum=0,
                    maximum=10,
                    step=1,
                    value=2,
                    interactive=True
                )
        with gr.Row():
            with gr.Column(1):
                provider = gr.Dropdown(
                    label="API Provider",
                    choices=[option.value for option in APIProvider],
                    value="openai"
                )
            with gr.Column(2):
                api_key = gr.Textbox(
                    label="API Key",
                    type="password",
                    value=state.value.get("api_key", ""),
                    placeholder="Paste your API key here",
                    interactive=True
                )

    # File Upload Section
    with gr.Accordion("File Upload & Management", open=True, elem_classes="accordion-header"):
        with gr.Row():
            with gr.Column():
                file_upload = gr.File(
                    label="Upload Files",
                    file_count="multiple",
                    type="filepath",
                    elem_classes="file-upload-area"
                )
            with gr.Column():
                with gr.Row():
                    upload_button = gr.Button("Upload Files", variant="primary", elem_classes="primary-button")
                    refresh_button = gr.Button("Refresh Files", variant="secondary", elem_classes="secondary-button")
        
        with gr.Row():
            # Initialize file choices as an empty list
            view_file_dropdown = gr.Dropdown(
                label="View File",
                choices=[],
                interactive=True
            )
            view_toggle = gr.Radio(
                label="Display Mode",
                choices=["OmniTool Computer", "File Viewer"],
                value="OmniTool Computer",
                interactive=True
            )

    with gr.Row():
        with gr.Column(scale=8):
            chat_input = gr.Textbox(
                show_label=False, 
                placeholder="Type a message to send to Omniparser + X ..."
            )
            submit_button = gr.Button(value="Submit", variant="primary")
        with gr.Column(scale=1, min_width=50):
            run_eval_button = gr.Button(value="Run Multi-Model Evaluation", variant="primary", elem_classes="primary-button")
        with gr.Column(scale=1, min_width=50):
            stop_button = gr.Button(value="Stop", variant="secondary", elem_classes="secondary-button")

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="Chatbot History", 
                autoscroll=True, 
                height=580,
                avatar_images=("👤", "🤖")
            )
        with gr.Column(scale=3):
            display_area = gr.HTML(
                get_file_viewer_html(),
                elem_classes="no-padding"
            )
            # Add card deck output for multi-model results
            card_deck_output = gr.HTML()

    # --- Multi-model + OmniParser agent logic ---
    def run_multi_model_eval(user_input, selected_models, evaluator, state):
        from functools import partial
        model_outputs = []
        for model_name in selected_models:
            # Setup state for each model
            temp_state = state.copy() if isinstance(state, dict) else dict(state)
            temp_state["model"] = model_name
            temp_state["messages"] = [{
                "role": Sender.USER,
                "content": [TextBlock(type="text", text=user_input)],
            }]
            temp_state['chatbot_messages'] = []
            # Run the agent for this model (single step, get output)
            output_text = None
            for loop_msg in sampling_loop_sync(
                model=model_name,
                provider=temp_state.get("provider", "openai"),
                messages=temp_state["messages"],
                output_callback=partial(chatbot_output_callback, chatbot_state=temp_state['chatbot_messages'], hide_images=True),
                tool_output_callback=partial(_tool_output_callback, tool_state=temp_state["tools"]),
                api_response_callback=partial(_api_response_callback, response_state=temp_state["responses"]),
                api_key=temp_state.get("api_key", os.environ.get("OPENAI_API_KEY", "")),
                only_n_most_recent_images=temp_state.get("only_n_most_recent_images", 2),
                max_tokens=2048,
                omniparser_url=args.omniparser_server_url,
                save_folder=str(RUN_FOLDER)
            ):
                # Only take the last output
                pass
            # Get the last assistant message as output
            for user_msg, bot_msg in temp_state['chatbot_messages'][::-1]:
                if bot_msg:
                    output_text = bot_msg
                    break
            if not output_text:
                output_text = "[No output]"
            model_outputs.append({
                "model": model_name,
                "output": output_text
            })
        # Evaluate all outputs
        evals = evaluate_model_outputs(model_outputs, evaluator)
        return render_model_cards(evals)

    run_eval_button.click(
        run_multi_model_eval,
        [chat_input, selected_models, evaluator, state],
        card_deck_output
    )
    
    only_n_images.change(fn=update_only_n_images, inputs=[only_n_images, state], outputs=None)
    provider.change(fn=update_provider, inputs=[provider, state], outputs=api_key)
    api_key.change(fn=update_api_key, inputs=[api_key, state], outputs=None)
    chatbot.clear(fn=clear_chat, inputs=[state], outputs=[chatbot])

    # File upload event handlers
    upload_button.click(
        fn=handle_file_upload,
        inputs=[file_upload, state],
        outputs=[view_file_dropdown]
    )
    
    # File viewing handlers
    view_file_dropdown.change(
        fn=view_file,
        inputs=[view_file_dropdown, view_toggle],
        outputs=[display_area]
    )
    
    submit_button.click(process_input, [chat_input, state], [chatbot, view_file_dropdown])
    stop_button.click(stop_app, [state], None)
    
    # Toggle view handler
    view_toggle.change(
        fn=toggle_view, 
        inputs=[view_toggle, view_file_dropdown, state], 
        outputs=[display_area, view_file_dropdown]
    )
    
    # Refresh files handler
    refresh_button.click(fn=refresh_files, inputs=[state], outputs=[view_file_dropdown])
    
    # Add JavaScript for auto-refresh instead of using demo.load()
    js_refresh = """
    function() {
        // Auto-refresh files every 5 seconds
        const refreshInterval = setInterval(function() {
            // Find and click the refresh button
            const refreshButtons = document.querySelectorAll('button');
            for (const button of refreshButtons) {
                if (button.textContent.includes('Refresh Files')) {
                    button.click();
                    break;
                }
            }
        }, 5000);
        
        // Return a cleanup function
        return () => clearInterval(refreshInterval);
    }
    """
    
    # Add the JavaScript to the page
    gr.HTML("<script>(" + js_refresh + ")();</script>")
    
# --- Evaluation logic stub ---
def evaluate_model_outputs(model_outputs, evaluator_model):
    """
    Given a list of model outputs and an evaluator model name, return a list of dicts:
    [{
        'model': <model_name>,
        'output': <model_output>,
        'evaluation': <short written evaluation>,
        'score': <numerical score>
    }, ...]
    """
    import re
    import json
    results = []
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set.")
    for mo in model_outputs:
        prompt = f"""
You are an expert model evaluator. Given the following model output, provide a short written evaluation (1-2 sentences) and a numerical score from 0 to 10 (10 = perfect, 0 = totally wrong).\n\nModel Output:\n{mo['output']}\n\nRespond in JSON with keys 'evaluation' and 'score'.
"""
        try:
            # For OpenAI Python SDK >=1.x
            response = openai.chat.completions.create(
                model=evaluator_model,
                messages=[
                    {"role": "system", "content": "You are a helpful and strict model evaluator."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=128,
                temperature=0.2,
                api_key=openai_api_key
            )
            content = response.choices[0].message.content
            # Try to extract JSON from the response
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
            else:
                # Fallback: try to parse as JSON directly
                parsed = json.loads(content)
            evaluation = parsed.get("evaluation", "No evaluation returned.")
            score = parsed.get("score", 0)
        except Exception as e:
            evaluation = f"[Error from evaluator: {e}]"
            score = 0
        results.append({
            'model': mo['model'],
            'output': mo['output'],
            'evaluation': evaluation,
            'score': score
        })
    return results
# --- End evaluation logic stub ---

if __name__ == "__main__":
    port = 7860  # Always use the same port
    demo.launch(server_name="0.0.0.0", server_port=port, share=True)

import gradio as gr

# List of 8-10 models (from Foundry catalog, example names)
MODEL_OPTIONS = [
    "gpt-4o", "gpt-4", "claude-3-opus", "claude-3-sonnet", "llama-3-70b", "qwen2.5vl", "deepseek-vl", "gemini-pro-vision", "mixtral-8x22b", "yi-34b-vl"
]

# List of 2-3 evaluator models
EVALUATOR_OPTIONS = [
    "gpt-4o-eval", "claude-3-opus-eval", "llama-3-70b-eval"
]

def run_models(user_input, selected_models):
    # Simulate model outputs for each selected model
    outputs = []
    for model in selected_models:
        outputs.append({
            "model": model,
            "output": f"[Stub output for '{model}']: {user_input[::-1]}"
        })
    return outputs

def evaluate_outputs(outputs, evaluator):
    # Simulate evaluation: short text + score for each model output
    results = []
    for o in outputs:
        results.append({
            "model": o["model"],
            "output": o["output"],
            "evaluation": f"[Stub] {evaluator} says: Output is plausible.",
            "score": 8.5
        })
    return results

def render_model_cards(evals):
    if not evals:
        return "<div style='text-align:center;color:#888;'>No model outputs to display yet.</div>"
    cards = []
    for e in evals:
        cards.append(f'''
        <div class="model-card">
            <h4>{e['model']}</h4>
            <div class="output"><b>Output:</b> {e['output']}</div>
            <div class="evaluation"><b>Evaluation:</b> {e['evaluation']}</div>
            <div class="score">Score: {e['score']}</div>
        </div>
        ''')
    return f'<div class="card-deck">{''.join(cards)}</div>'

def main(user_input, selected_models, evaluator):
    outputs = run_models(user_input, selected_models)
    evals = evaluate_outputs(outputs, evaluator)
    return render_model_cards(evals)

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate")) as demo:
    gr.Markdown("""
    <div style='text-align:center;margin-bottom:16px;'>
        <h2 style='margin-bottom:0.2em;color:#2d3748;'>Multi-Model Output Evaluator</h2>
        <p style='color:#4a5568;'>Prototype: Select models, an evaluator, and see results in a beautiful card deck.</p>
    </div>
    """)
    with gr.Row():
        user_input = gr.Textbox(label="Input Prompt", placeholder="Enter your prompt here...", elem_classes="input-prompt")
    with gr.Row():
        selected_models = gr.CheckboxGroup(MODEL_OPTIONS, label="Select Models", value=[MODEL_OPTIONS[0]], elem_classes="model-select")
        evaluator = gr.Dropdown(EVALUATOR_OPTIONS, label="Evaluator Model", value=EVALUATOR_OPTIONS[0], elem_classes="evaluator-select")
    run_btn = gr.Button("Run Evaluation", variant="primary", elem_id="run-btn")
    output = gr.HTML()
    run_btn.click(main, [user_input, selected_models, evaluator], output)
    gr.HTML("""
    <style>
    .input-prompt input {font-size:1.1em;padding:8px;}
    .model-select label, .evaluator-select label {font-weight:500;}
    #run-btn {background:#2563eb;color:#fff;border-radius:6px;font-weight:600;box-shadow:0 1px 4px #0001;}
    #run-btn:hover {background:#1d4ed8;}
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

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)

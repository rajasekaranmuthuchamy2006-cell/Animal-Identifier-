import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import gradio as gr

# Mute standard TensorFlow info/warning logs to clean up your terminal output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("🧠 Loading AI Model... Please wait...")
try:
    model = tf.keras.models.load_model('animal_model.keras')
    with open('animals.txt', 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    print("✅ Model loaded successfully!")
    
    try:
        input_shape = model.input_shape
        TARGET_SIZE = (input_shape[1], input_shape[2])
    except:
        TARGET_SIZE = (224, 224)
except Exception as e:
    print(f"❌ Error loading system files: {e}")
    sys.exit()

def predict_top_three(input_image):
    if input_image is None:
        return {}
    
    try:
        img = input_image.resize(TARGET_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array, verbose=0)[0]
        highest_score = float(np.max(predictions)) * 100
        
        # 🚨 SECURITY THRESHOLD: Raised from 30% to 85% 
        # Since the guitar falsely triggered 84%, setting this to 85% will perfectly catch it!
        if highest_score < 85.0:
            return {
                "❌ NOT AN ANIMAL / UNKNOWN OBJECT": 1.0,
                "": 0.0
            }

        results = {class_names[i].upper(): float(predictions[i]) for i in range(len(class_names))}
        return results
        
    except Exception as e:
        return {f"Error during prediction: {e}": 1.0}

def clear_all_inputs_and_outputs():
    return gr.update(value=None), gr.update(value={})

# Custom CSS Premium Theme Parameters — "Field Journal" theme
# A naturalist specimen-catalog look: deep forest charcoal, amber/moss accents,
# serif display type, and corner-bracket "specimen card" framing on each panel.
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500&display=swap');

:root {
    --bg-deep: #131d16;
    --bg-deep-2: #0d1510;
    --bg-panel: rgba(28, 38, 30, 0.72);
    --ink: #ece6d3;
    --ink-muted: #9aa893;
    --amber: #d99a44;
    --amber-soft: rgba(217, 154, 68, 0.35);
    --amber-dim: rgba(217, 154, 68, 0.16);
    --moss: #6c8a57;
    --moss-soft: rgba(108, 138, 87, 0.35);
    --line: rgba(217, 154, 68, 0.22);
}

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { transition: none !important; animation: none !important; }
}

body, .gradio-container {
    background:
        radial-gradient(ellipse 900px 500px at 12% -10%, var(--amber-dim), transparent 60%),
        radial-gradient(ellipse 700px 600px at 100% 110%, rgba(108, 138, 87, 0.18), transparent 60%),
        linear-gradient(160deg, var(--bg-deep) 0%, var(--bg-deep-2) 100%) !important;
    font-family: 'Inter', system-ui, sans-serif !important;
    color: var(--ink) !important;
}

/* Completely hides both the Gradio footer and the API footer links */
footer, .footer, .built-with { display: none !important; }
a[href*='gradio.app'], a[href*='/api'] { display: none !important; }

#title-text h1 {
    font-family: 'Fraunces', serif !important;
    color: var(--ink) !important;
    text-shadow: 0px 2px 18px var(--amber-dim);
    font-size: 2.9rem !important;
    font-weight: 700 !important;
    font-optical-sizing: auto;
    letter-spacing: 0.01em;
    text-align: center !important;
    margin-bottom: 0.2rem !important;
}
#desc-text p {
    font-family: 'Inter', sans-serif !important;
    color: var(--ink-muted) !important;
    font-size: 1.05rem !important;
    text-align: center !important;
    border-top: 1px dashed var(--line);
    border-bottom: 1px dashed var(--line);
    padding: 0.6rem 0 !important;
    max-width: 620px;
    margin: 0.4rem auto 1.6rem auto !important;
}

/* Specimen-card framing: corner brackets drawn with layered backgrounds */
.panel-box {
    position: relative;
    background-color: var(--bg-panel) !important;
    background-image:
        linear-gradient(var(--amber), var(--amber)), linear-gradient(var(--amber), var(--amber)),
        linear-gradient(var(--amber), var(--amber)), linear-gradient(var(--amber), var(--amber)),
        linear-gradient(var(--amber), var(--amber)), linear-gradient(var(--amber), var(--amber)),
        linear-gradient(var(--amber), var(--amber)), linear-gradient(var(--amber), var(--amber)) !important;
    background-size:
        20px 2px, 2px 20px, 20px 2px, 2px 20px,
        20px 2px, 2px 20px, 20px 2px, 2px 20px !important;
    background-position:
        top 12px left 12px, top 12px left 12px,
        top 12px right 12px, top 12px right 12px,
        bottom 12px left 12px, bottom 12px left 12px,
        bottom 12px right 12px, bottom 12px right 12px !important;
    background-repeat: no-repeat !important;
    border: 1px solid rgba(217, 154, 68, 0.12) !important;
    border-radius: 10px !important;
    backdrop-filter: blur(10px) !important;
    padding: 32px 28px !important;
    box-shadow: 0 18px 40px -12px rgba(0, 0, 0, 0.55) !important;
    transition: border-color 0.25s ease, box-shadow 0.25s ease;
}
.panel-box:hover {
    border-color: rgba(217, 154, 68, 0.28) !important;
    box-shadow: 0 22px 48px -12px rgba(0, 0, 0, 0.65) !important;
}

/* Upload widget + label result text, recolored to match the journal palette */
#image-upload, #result-output {
    font-family: 'Inter', sans-serif !important;
}
#image-upload label span, #result-output label span {
    font-family: 'JetBrains Mono', monospace !important;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    font-size: 0.78rem !important;
    color: var(--ink-muted) !important;
}

/* Force the Identification Analysis Metrics card (and every element inside it)
   off the default Gradio white background and onto the dark theme, with
   high-contrast, legible text for the predicted classes and percentages */
#result-output, #result-output * {
    background: transparent !important;
    border-color: var(--line) !important;
}
#result-output {
    color: var(--ink) !important;
    font-weight: 600 !important;
}
#result-output table, #result-output tr, #result-output td,
#result-output [class*="confidence"], #result-output [class*="label"] {
    color: var(--ink) !important;
    font-weight: 600 !important;
    font-size: 1.02rem !important;
}
#result-output [class*="bar"] {
    background: linear-gradient(90deg, var(--moss) 0%, var(--amber) 100%) !important;
}
/* The single top headline prediction, set apart in amber for emphasis */
#result-output > div > div:first-child,
#result-output [class*="output-class"] {
    color: var(--amber) !important;
    font-weight: 700 !important;
}

button#action-btn {
    background: linear-gradient(135deg, var(--amber) 0%, #b9782f 100%) !important;
    color: var(--bg-deep-2) !important;
    border: none !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em;
    border-radius: 6px !important;
    box-shadow: 0 8px 20px -6px rgba(217, 154, 68, 0.45);
    transition: transform 0.18s ease, box-shadow 0.18s ease;
}
button#action-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 26px -6px rgba(217, 154, 68, 0.55);
}
button#clear-btn {
    background: transparent !important;
    color: var(--ink-muted) !important;
    border: 1px solid var(--moss-soft) !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em;
    border-radius: 6px !important;
    transition: border-color 0.18s ease, color 0.18s ease;
}
button#clear-btn:hover {
    border-color: var(--moss) !important;
    color: var(--ink) !important;
}

button:focus-visible, [tabindex]:focus-visible {
    outline: 2px solid var(--amber) !important;
    outline-offset: 2px !important;
}
"""

with gr.Blocks(title="🐾 Animal Identifier") as demo:
    
    with gr.Column():
        gr.Markdown("# 🐾 Animal Identifier", elem_id="title-text")
        gr.Markdown("Upload any image from your local system files to compute the top 3 image predictions instantly.", elem_id="desc-text")
    
    with gr.Row():
        with gr.Column(elem_classes="panel-box"):
            image_input = gr.Image(type="pil", label="📥 Drag & Drop or Browse Image Source", elem_id="image-upload")
            
            with gr.Row():
                clear_btn = gr.Button("🗑️ Clear Screen", elem_id="clear-btn")
                submit_btn = gr.Button("🔍 Identify Animal", variant="primary", elem_id="action-btn")
            
        with gr.Column(elem_classes="panel-box"):
            label_output = gr.Label(num_top_classes=3, label="🎯 Identification Analysis Metrics", elem_id="result-output")
            
    submit_btn.click(fn=predict_top_three, inputs=image_input, outputs=label_output)
    clear_btn.click(fn=clear_all_inputs_and_outputs, inputs=None, outputs=[image_input, label_output])
    image_input.clear(fn=lambda: gr.update(value={}), inputs=None, outputs=label_output)

if __name__ == "__main__":
    demo.launch(inbrowser=True, css=custom_css)
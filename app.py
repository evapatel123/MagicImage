import torch
import spaces
import gradio as gr
from diffusers import DiffusionPipeline

# ============================================================
# Load the pipeline once at startup
# ============================================================

print("Loading Z-Image-Turbo pipeline...")

pipe = DiffusionPipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
)

pipe.to("cuda")

# ============================================================
# Optional AoTI + Flash Attention 3
# ============================================================

# pipe.transformer.layers._repeated_blocks = ["ZImageTransformerBlock"]
# spaces.aoti_blocks_load(
#     pipe.transformer.layers,
#     "zerogpu-aoti/Z-Image",
#     variant="fa3"
# )

print("Pipeline loaded successfully!")

# ============================================================
# Image Generation Function
# ============================================================

@spaces.GPU
def generate_image(
    prompt,
    height,
    width,
    num_inference_steps,
    seed,
    randomize_seed,
    progress=gr.Progress(track_tqdm=True),
):
    """Generate an AI image from a text prompt."""

    if randomize_seed:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()

    generator = torch.Generator("cuda").manual_seed(int(seed))

    image = pipe(
        prompt=prompt,
        height=int(height),
        width=int(width),
        num_inference_steps=int(num_inference_steps),
        guidance_scale=0.0,
        generator=generator,
    ).images[0]

    return image, seed


# ============================================================
# Example Prompts
# ============================================================

examples = [
    [ 
    "Cyberpunk city at night, neon skyscrapers reflecting on wet streets, futuristic vehicles flying between buildings, cinematic lighting, ultra detailed, 8K digital art"
    ],
    
    
    [
    "A majestic white tiger walking through a mystical forest, glowing plants, foggy atmosphere, fantasy concept art, extremely detailed"
    ],
    
    
    [
    "Luxury futuristic mansion on a cliff overlooking the ocean, sunset lighting, modern architecture, cinematic photography, realistic"
    ],
    
    
    [
    "Ancient Japanese temple surrounded by cherry blossoms during snowfall, peaceful atmosphere, soft lighting, highly detailed landscape photography"
    ],
    
    
    [
    "Portrait of a futuristic astronaut explorer wearing advanced armor, reflections on helmet, galaxy background, cinematic sci-fi art"
    ],
]

# ============================================================
# Build the Gradio Interface
# ============================================================

with gr.Blocks(
    theme=gr.themes.Ocean(),
    fill_height=True,
    elem_id="ai-studio"
) as demo:

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    gr.Markdown(
        """
# 🌊 MagicImage

### Create stunning AI artwork from your imagination**

Generate high-quality images with advanced AI technology. Transform simple text prompts into **detailed illustrations**, **cinematic scenes**, **realistic portraits**, and creative masterpieces **in seconds.**.
""",
        elem_classes="header-text",
    )

    with gr.Row(equal_height=False):

        # ====================================================
        # LEFT COLUMN
        # ====================================================

        with gr.Column(scale=1, min_width=320):

            prompt = gr.Textbox(
                label="✨ Prompt",
                placeholder="Describe the image you'd like to generate...",
                lines=5,
                max_lines=10,
                autofocus=True,
            )

            with gr.Accordion(
                "⚙️ Advanced Settings",
                open=False,
            ):

                with gr.Row():

                    height = gr.Slider(
                        minimum=512,
                        maximum=2048,
                        value=1024,
                        step=64,
                        label="Height",
                        info="Image height in pixels",
                    )

                    width = gr.Slider(
                        minimum=512,
                        maximum=2048,
                        value=1024,
                        step=64,
                        label="Width",
                        info="Image width in pixels",
                    )

                num_inference_steps = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=9,
                    step=1,
                    label="Inference Steps",
                    info="Recommended: 9",
                )

                with gr.Row():

                    randomize_seed = gr.Checkbox(
                        label="🎲 Random Seed",
                        value=True,
                    )

                    seed = gr.Number(
                        label="Seed",
                        value=42,
                        precision=0,
                        visible=False,
                    )

                def toggle_seed(randomize):
                    return gr.Number(
                        visible=not randomize
                    )

                randomize_seed.change(
                    toggle_seed,
                    inputs=randomize_seed,
                    outputs=seed,
                )

            generate_btn = gr.Button(
                "🚀 Generate Image",
                variant="primary",
                size="lg",
            )

            gr.Examples(
                examples=examples,
                inputs=prompt,
                label="💡 Example Prompts",
                examples_per_page=5,
            )

                    # ====================================================
        # RIGHT COLUMN
        # ====================================================

        with gr.Column(scale=1, min_width=320):

            output_image = gr.Image(
                label="Generated Image",
                type="pil",
                format="png",
                show_label=False,
                height=600,
            )

            used_seed = gr.Number(
                label="🎲 Seed Used",
                interactive=False,
                container=True,
            )


    # ========================================================
    # Footer
    # ========================================================

    gr.Markdown(
        """
---
<div class="footer-text">

<strong>Model:</strong>
<a href="https://huggingface.co/Tongyi-MAI/Z-Image-Turbo" target="_blank">
Tongyi-MAI/Z-Image-Turbo
</a>

<br>

<strong>Made by</strong>
<a href="https://github.com/evapatel123" target="_blank">
evapatel123
</a>

<br>

<strong>Copyrighted</strong>
<a href="https://github.com/evapatel123" target="_blank">
© 2026 Eva Patel, Some Rights Reserved
</a>

</div>
""",
        elem_classes="footer-text",
    )


    # ========================================================
    # Button Event
    # ========================================================

    generate_btn.click(
        fn=generate_image,
        inputs=[
            prompt,
            height,
            width,
            num_inference_steps,
            seed,
            randomize_seed,
        ],
        outputs=[
            output_image,
            used_seed,
        ],
    )


    # ========================================================
    # Press Enter To Generate
    # ========================================================

    prompt.submit(
        fn=generate_image,
        inputs=[
            prompt,
            height,
            width,
            num_inference_steps,
            seed,
            randomize_seed,
        ],
        outputs=[
            output_image,
            used_seed,
        ],
    )


# ============================================================
# Launch Application
# ============================================================

if __name__ == "__main__":

    demo.launch(
        css="""
/* ==========================================
   Animated AI Background
========================================== */

body {
    background:
    linear-gradient(
        120deg,
        #0f172a,
        #172554,
        #0f172a
    );

    background-size: 300% 300%;

    animation:
    backgroundMove 12s ease infinite;
}


@keyframes backgroundMove {

    0% {
        background-position:0% 50%;
    }

    50% {
        background-position:100% 50%;
    }

    100% {
        background-position:0% 50%;
    }

}



/* ==========================================
   Main Container
========================================== */


.gradio-container {

    max-width:
    1400px !important;

    margin:
    auto !important;

}



/* ==========================================
   Animated Header
========================================== */


.header-text h1 {

    font-size:
    3rem !important;

    font-weight:
    900 !important;

    background:
    linear-gradient(
        90deg,
        #38bdf8,
        #6366f1,
        #a855f7,
        #38bdf8
    );

    background-size:
    300%;

    animation:
    gradientText 6s linear infinite;

    -webkit-background-clip:
    text;

    -webkit-text-fill-color:
    transparent;

}



@keyframes gradientText {

    0% {
        background-position:
        0%;
    }

    100% {
        background-position:
        300%;
    }

}



/* ==========================================
   Glass Cards
========================================== */


.gr-box,
.gr-panel,
.gr-group {

    background:
    rgba(
        255,
        255,
        255,
        0.08
    ) !important;


    backdrop-filter:
    blur(20px);


    border-radius:
    24px !important;


    border:
    1px solid
    rgba(
        255,
        255,
        255,
        .15
    ) !important;


    animation:
    fadeUp .8s ease;

}



@keyframes fadeUp {

    from {

        opacity:
        0;

        transform:
        translateY(25px);

    }


    to {

        opacity:
        1;

        transform:
        translateY(0);

    }

}



/* ==========================================
   Generate Button
========================================== */


button.primary {


    background:
    linear-gradient(
        135deg,
        #06b6d4,
        #2563eb,
        #9333ea
    ) !important;


    background-size:
    200%;


    animation:
    buttonGradient 4s infinite;


    border-radius:
    18px !important;


    font-size:
    18px !important;


    font-weight:
    800 !important;


    color:
    white !important;


    border:
    none !important;


    transition:
    .3s ease;

}



@keyframes buttonGradient {

    0% {
        background-position:
        0%;
    }


    50% {
        background-position:
        100%;
    }


    100% {
        background-position:
        0%;
    }

}



button.primary:hover {


    transform:
    translateY(-5px)
    scale(1.03);


    box-shadow:

    0 0 35px
    rgba(
        56,
        189,
        248,
        .7
    );

}



/* ==========================================
   Textbox Animation
========================================== */


textarea,
input {


    border-radius:
    18px !important;


    transition:
    .3s ease;

}



textarea:focus,
input:focus {


    transform:
    scale(1.02);


    box-shadow:

    0 0 20px
    rgba(
        56,
        189,
        248,
        .4
    );

}



/* ==========================================
   Image Animation
========================================== */


.image-container {


    border-radius:
    25px !important;


    overflow:
    hidden;

}



.image-container img {


    transition:
    .5s ease;

}



.image-container img:hover {


    transform:
    scale(1.05);

}



/* ==========================================
   Example Cards
========================================== */


.gr-example {


    transition:
    .3s ease;


}



.gr-example:hover {


    transform:
    translateY(-8px);


    box-shadow:

    0 15px 30px
    rgba(
        0,
        0,
        0,
        .25
    );

}



/* ==========================================
   Footer
========================================== */


.footer-text {


    opacity:
    .75;


    animation:
    fadeUp 1.2s ease;

}


.footer-text a {


    color:
    #38bdf8 !important;


    transition:
    .3s;

}


.footer-text a:hover {


    color:
    #a855f7 !important;

}



/* ==========================================
   Loading Animation
========================================== */


.generating {


    animation:
    pulse 1.5s infinite;

}



@keyframes pulse {


    0% {

        opacity:
        1;

    }


    50% {

        opacity:
        .5;

    }


    100% {

        opacity:
        1;

    }

}


/* ======================================
   Glass Panels
====================================== */


.gr-panel,
.gr-box,
.gr-group {


    background:

    rgba(
        255,
        255,
        255,
        .08
    ) !important;


    backdrop-filter:
    blur(25px);


    border:

    1px solid

    rgba(
        255,
        255,
        255,
        .15
    ) !important;


    box-shadow:

    0 20px 50px

    rgba(
        0,
        0,
        0,
        .25
    );


}



/* ======================================
   Animated Header
====================================== */


.header-text h1 {


    font-size:
    3rem !important;


    font-weight:
    900 !important;


    background:

    linear-gradient(
        90deg,
        #38bdf8,
        #818cf8,
        #c084fc,
        #38bdf8
    );


    background-size:
    300%;


    animation:

    textGlow 5s infinite linear;


    -webkit-background-clip:
    text;


    color:
    transparent !important;

}



@keyframes textGlow {


    from {

        background-position:
        0%;

    }


    to {

        background-position:
        300%;

    }

}



/* ======================================
   Generate Button Neon Effect
====================================== */


button.primary {


    position:
    relative;


    overflow:
    hidden;


    background:

    linear-gradient(
        135deg,
        #06b6d4,
        #2563eb,
        #9333ea
    ) !important;


    border-radius:
    20px !important;


    font-weight:
    800 !important;


    color:
    white !important;


    box-shadow:

    0 0 25px

    rgba(
        56,
        189,
        248,
        .5
    );


}



button.primary:hover {


    transform:

    translateY(-5px)

    scale(1.05);


    box-shadow:

    0 0 50px

    rgba(
        168,
        85,
        247,
        .8
    );

}



/* ======================================
   Moving Button Shine
====================================== */


button.primary::before {


    content:"";


    position:absolute;


    top:0;


    left:-100%;


    width:50%;


    height:100%;


    background:

    linear-gradient(
        120deg,
        transparent,
        rgba(255,255,255,.6),
        transparent
    );


    animation:

    shine 3s infinite;

}



@keyframes shine {


    100% {

        left:150%;

    }

}



/* ======================================
   Image Animation
====================================== */


.image-container {


    border-radius:
    25px !important;


    overflow:
    hidden;

}



.image-container img {


    animation:

    imageAppear .8s ease;


}



@keyframes imageAppear {


    from {


        opacity:
        0;


        transform:

        scale(.9)

        translateY(30px);

    }


    to {


        opacity:
        1;


        transform:

        scale(1)

        translateY(0);

    }

}



/* ======================================
   Textbox Glow
====================================== */


textarea:focus {


    border:

    1px solid

    #38bdf8 !important;


    box-shadow:

    0 0 25px

    rgba(
        56,
        189,
        248,
        .4
    );

}



/* ======================================
   Example Cards
====================================== */


.gr-example {


    transition:
    .3s ease;


}



.gr-example:hover {


    transform:

    translateY(-10px);


    box-shadow:

    0 15px 40px

    rgba(
        0,
        0,
        0,
        .4
    );

}



/* ======================================
   Smooth Entrance
====================================== */


.gradio-container > * {


    animation:

    fadeIn 1s ease;

}



@keyframes fadeIn {


    from {

        opacity:0;

        transform:
        translateY(20px);

    }


    to {

        opacity:1;

        transform:
        translateY(0);

    }

}



        """,

        footer_links=[
            "api",
            "gradio"
        ],

        mcp_server=True,
    )
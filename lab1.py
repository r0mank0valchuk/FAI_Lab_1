import gradio as gr
import numpy as np
import cv2
from PIL import Image, ImageDraw
import io

def analyze_image(image, grid="5x5", norm_type="За сумою"):
    gray = image.convert("L")
    img_array = np.array(gray)

    _, binary = cv2.threshold(img_array, 128, 255, cv2.THRESH_BINARY)
    h, w = binary.shape

    rows, cols = map(int, grid.split("x"))
    cell_h, cell_w = h // rows, w // cols

    abs_vector = []

    draw_img = Image.fromarray(cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB))
    draw = ImageDraw.Draw(draw_img)

    for i in range(rows):
        for j in range(cols):
            y1, y2 = i * cell_h, (i + 1) * cell_h if i < rows - 1 else h
            x1, x2 = j * cell_w, (j + 1) * cell_w if j < cols - 1 else w

            cell = binary[y1:y2, x1:x2]
            black_pixels = np.count_nonzero(cell == 0)
            abs_vector.append(int(black_pixels))

            if j > 0:
                draw.line([(x1, 0), (x1, h)], fill=(255, 0, 0), width=1)
            if i > 0:
                draw.line([(0, y1), (w, y1)], fill=(255, 0, 0), width=1)

    draw.rectangle([(0, 0), (w - 1, h - 1)], outline=(0, 255, 0), width=2)

    abs_array = np.array(abs_vector, dtype=float)
    if norm_type == "За сумою":
        total = np.sum(abs_array)
        norm_vector = (abs_array / total).tolist() if total > 0 else [0] * len(abs_array)
    else: 
        max_val = np.max(abs_array)
        norm_vector = (abs_array / max_val).tolist() if max_val > 0 else [0] * len(abs_array)

    abs_text = "; ".join(str(int(v)) for v in abs_vector)
    norm_text = "; ".join(f"{v:.6f}" for v in norm_vector)

    return draw_img, abs_text, norm_text

with gr.Blocks(title="AI_LAB_1") as demo:
    gr.Markdown("# Lab 1")

    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(label="завантажте зображення", type="pil")
            grid_input = gr.Radio(
                ["3x3", "4x4", "5x5", "6x6", "4x5", "5x4"],
                value="5x5",
                label="розмір сітки"
            )
            norm_choice = gr.Radio(
                ["За сумою", "За максимумом"],
                value="За сумою",
                label="Тип нормування"
            )
            analyze_btn = gr.Button("run")

        with gr.Column(scale=2):
            img_output = gr.Image(label="розмічене зображення (з сіткою)")
            abs_output = gr.Textbox(label="абсолютний вектор ознак", lines=5)
            norm_output = gr.Textbox(label="нормований вектор ознак", lines=5)

    analyze_btn.click(
        fn=analyze_image,
        inputs=[img_input, grid_input, norm_choice],
        outputs=[img_output, abs_output, norm_output]
    )

demo.launch()


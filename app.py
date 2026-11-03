
import os
import time
import torch
import gradio as gr

from torchvision import transforms
from PIL import Image
from model import ImprovedSteelDefectCNN


# 1. Configure Device ================================


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)



# 2. Define Defect Categories ========================


class_names = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]


# 3. Load Trained CNN ===============================

model = ImprovedSteelDefectCNN().to(device)

model_path = os.path.join(
    os.path.dirname(__file__),
    "steel_defect_cnn_improved.pth"
)

state_dict = torch.load(
    model_path,
    map_location=device,
    weights_only=True
)

model.load_state_dict(state_dict)
model.eval()


# 4. Image Preprocessing ===========================

prediction_transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# 5. Prediction Function ==========================

def predict_steel_defect(image):

    if image is None:
        return {}, "Please upload an image."

    image = image.convert("RGB")

    image_tensor = prediction_transform(
        image
    ).unsqueeze(0).to(device)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    with torch.inference_mode():

        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[0]

    if device.type == "cuda":
        torch.cuda.synchronize()

    end_time = time.perf_counter()

    inference_time = (end_time - start_time) * 1000

    results = {
        class_names[i]: float(probabilities[i].item())
        for i in range(len(class_names))
    }

    predicted_defect = max(
        results,
        key=results.get
    )

    formatted_defect = predicted_defect.replace(
        "_", " "
    ).title()

    information = (
        f"Predicted Defect: {formatted_defect}\n"
        f"Inference Time: {inference_time:.2f} ms\n"
        f"Processing Device: {device}"
    )

    return results, information


# 6. Gradio Interface ==========================

demo = gr.Interface(

    fn=predict_steel_defect,

    inputs=gr.Image(
        type="pil",
        label="Upload Steel Surface Image"
    ),

    outputs=[
        gr.Label(
            num_top_classes=6,
            label="Defect Classification"
        ),

        gr.Textbox(
            label="Prediction Information",
            lines=3
        )
    ],

    title="AI Visual Quality Inspector",

    description=(
        "Upload a steel surface image to classify "
        "one of six surface defect categories using "
        "a custom-trained PyTorch CNN. "
        "This model does not identify defect-free surfaces."
    ),

    flagging_mode="never"
)


# 7. Launch Application ================================

if __name__ == "__main__":
    demo.launch()

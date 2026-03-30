"""Check if Qwen3.5 vision features have square spatial structure."""
import torch
import math
from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor
from PIL import Image
import numpy as np

# Load model
model_id = "Qwen/Qwen3.5-0.8B"  # 根据你的配置调整
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)
processor = AutoProcessor.from_pretrained(model_id)

# Create dummy image
image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

# Process
messages = [[{
    "role": "user",
    "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "test"}
    ]
}]]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
).to("cuda")

# Extract vision features
with torch.no_grad():
    pixel_values = inputs['pixel_values']
    image_grid_thw = inputs['image_grid_thw']

    vision_output = model.model.visual(
        pixel_values.type(model.model.visual.dtype),
        grid_thw=image_grid_thw,
        return_dict=True
    )
    vision_feats = vision_output.last_hidden_state

# Check shape
B, N, D = vision_feats.shape
H = int(math.sqrt(N))

print(f"Vision features shape: {vision_feats.shape}")
print(f"Token count (N): {N}")
print(f"sqrt(N): {math.sqrt(N)}")
print(f"int(sqrt(N)): {H}")
print(f"H × H: {H * H}")
print(f"Is perfect square: {H * H == N}")

if H * H != N:
    print(f"\n⚠️ WARNING: {N} is NOT a perfect square!")
    print(f"BottleneckSE will FAIL with this token count.")
else:
    print(f"\n✅ OK: {N} = {H}×{H} is a perfect square.")

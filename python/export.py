"""
Train a tiny MLP on MNIST, then export its weights + one test image as raw
float32 binary files for the C++ engine to load.

Run from the PROJECT ROOT so that `model/` lands next to src/ :
    python python/export.py
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms


# --- model: 784 -> 128 -> 10 (C++ must use the SAME architecture) ---
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)          # flatten 28x28 -> 784
        x = F.relu(self.fc1(x))      # Linear -> ReLU
        x = self.fc2(x)              # Linear -> raw scores (logits)
        return x                     # no softmax here; applied at inference


# --- data: ToTensor() scales pixels to [0, 1] ---
transform = transforms.ToTensor()
train_ds = datasets.MNIST("data", train=True, download=True, transform=transform)
test_ds = datasets.MNIST("data", train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)

# --- train a few epochs on CPU (seconds-ish) ---
model = MLP()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(3):
    for x, y in train_loader:
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
    print(f"epoch {epoch} done")
model.eval()

# --- export every array as raw float32, row-major (numpy default) ---
os.makedirs("model", exist_ok=True)


def save(arr, name):
    arr = np.ascontiguousarray(arr, dtype=np.float32)   # row-major float32
    arr.tofile(f"model/{name}.bin")
    print(f"saved model/{name}.bin  shape={arr.shape}")


# NOTE: PyTorch nn.Linear stores weight as [out, in] and computes x @ W.T.
# Our C++ matmul does input[1 x in] * W[in x out], so we TRANSPOSE (.T) here
# to export W as [in, out]. Get this wrong and the result is garbage.
save(model.fc1.weight.detach().numpy().T, "fc1_weight")   # [784, 128]
save(model.fc1.bias.detach().numpy(),     "fc1_bias")     # [128]
save(model.fc2.weight.detach().numpy().T, "fc2_weight")   # [128, 10]
save(model.fc2.bias.detach().numpy(),     "fc2_bias")     # [10]

# --- export one test image (already in [0,1]) + print the answer to check C++ against ---
img, label = test_ds[0]
save(img.numpy().reshape(784), "test_image")              # [784]
with torch.no_grad():
    logits = model(img.unsqueeze(0))
    pred = int(logits.argmax(1).item())
print(f"\ntest image true label = {label}")
print(f"pytorch prediction    = {pred}")
print(f"pytorch logits        = {logits.numpy().reshape(-1)}")

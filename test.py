import torch
import matplotlib.pyplot as plt
from utils import CorrEncoder

# Load data and model
data = torch.load('processed_data_bidmc/bidmc_all.pt')
model = CorrEncoder()
checkpoint = torch.load('capnobase_master.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Get Subject 01 data (first 50 windows)
ppg = data['samples'][50:100].unsqueeze(1) # [50, 1, 288]
resp = data['labels'][50:100].reshape(-1).numpy() # Ground Truth

# Inference
with torch.no_grad():
    pred = model(ppg).squeeze(1).reshape(-1).numpy() # [14400]

# Plot
plt.figure(figsize=(15, 5))
plt.plot(resp[:1000], label='Ground Truth (Impedance)', alpha=0.7)
plt.plot(pred[:1000], label='Prediction (Zero-shot)', alpha=0.7)
plt.title('Subject 01: First 1000 samples')
plt.legend()
plt.savefig('debug_zeroshot.png')
print("Saved debug_zeroshot.png")
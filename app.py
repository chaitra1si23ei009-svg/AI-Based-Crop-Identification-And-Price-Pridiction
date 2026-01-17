import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn

# ============================================================
# Crop Name Mapping (Excel-friendly names)
# ============================================================
crop_map = {
    "rice": "Sona Masoori Rice",
    "maize": "Maize",
    "wheat": "Wheat"
}

# ============================================================
# Streamlit Title
# ============================================================
st.title("🌾 Crop Identification & Price Forecast")
st.write("Upload or capture a real image of Maize, Rice, or Wheat")

# ============================================================
# Load Pretrained ResNet50 (Cached)
# ============================================================
@st.cache_resource
def load_model():
    import os
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3)

    checkpoint_path = os.path.join(os.path.dirname(__file__), "resnet_crop_model_with_mapping.pth")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    idx_to_class = checkpoint['idx_to_class']
    return model, idx_to_class

model, idx_to_class = load_model()

# ============================================================
# Input Method: Upload OR Webcam
# ============================================================
choice = st.radio("Input Method:", ["Upload Image", "Use Webcam"])

image_file = None
if choice == "Upload Image":
    image_file = st.file_uploader("Upload Crop Image", type=["jpg","jpeg","png"])
elif choice == "Use Webcam":
    image_file = st.camera_input("Capture Crop Image")

# ============================================================
# If Image Exists → Predict Crop
# ============================================================
if image_file is not None:
    img = Image.open(image_file).convert("RGB")
    st.image(img, caption="Input Image", use_column_width=True)

    # Transform
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
    img_tensor = transform(img).unsqueeze(0)

    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        _, pred_idx = torch.max(output, 1)
        predicted_crop = idx_to_class[pred_idx.item()]
        excel_crop = crop_map[predicted_crop]

    st.success(f"Predicted Crop: **{predicted_crop}** → Excel Name: **{excel_crop}**")

    # ============================================================
    # Load Price Excel
    # ============================================================
    import os
    excel_path = os.path.join(os.path.dirname(__file__), "price.xlsx.xlsx")
    df = pd.read_excel(excel_path)
    crop_df = df[df["Crop"] == excel_crop].sort_values("Date")

    if len(crop_df) < 4:
        st.warning("Not enough historical data to train LSTM for forecasting.")
        st.stop()

    # ============================================================
    # Prepare Data for LSTM
    # ============================================================
    prices = crop_df["Price (INR/quintal)"].values.astype(float)

    min_price, max_price = prices.min(), prices.max()
    prices_norm = (prices - min_price) / (max_price - min_price)

    seq_length = 3
    X, Y = [], []
    for i in range(len(prices_norm) - seq_length):
        X.append(prices_norm[i:i+seq_length])
        Y.append(prices_norm[i+seq_length])

    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
    Y = torch.tensor(Y, dtype=torch.float32).unsqueeze(-1)

    # ============================================================
    # LSTM Model (FIXED __init__)
    # ============================================================
    class PriceLSTM(nn.Module):
        def __init__(self, input_size=1, hidden_size=50, num_layers=1, output_size=1):
            super(PriceLSTM, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
            out, _ = self.lstm(x, (h0, c0))
            out = self.fc(out[:, -1, :])
            return out

    lstm = PriceLSTM()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=0.01)

    # ============================================================
    # Train LSTM
    # ============================================================
    st.subheader("Identifying crop and predicting  crop price")
    progress_bar = st.progress(0)

    epochs = 200
    for epoch in range(epochs):
        lstm.train()
        output_train = lstm(X)
        loss = criterion(output_train, Y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress_bar.progress(int((epoch+1)/epochs * 100))

    st.success("predicting crop and price")

    # ============================================================
    # Forecast Next 2 Months
    # ============================================================
    lstm.eval()
    last_seq = X[-1].unsqueeze(0)
    forecast = []

    for _ in range(2):
        next_price = lstm(last_seq).item()
        forecast.append(next_price)

        next_seq = last_seq.squeeze(0).detach().numpy()
        next_seq = np.append(next_seq[1:], [[next_price]], axis=0)
        last_seq = torch.tensor(next_seq, dtype=torch.float32).unsqueeze(0)

    # Denormalize
    forecast_denorm = [p * (max_price - min_price) + min_price for p in forecast]

    # ============================================================
   

    st.success(f"Predicted Prices:\n- Month 1 → ₹{forecast_denorm[0]:.2f}\n- Month 2 → ₹{forecast_denorm[1]:.2f}")

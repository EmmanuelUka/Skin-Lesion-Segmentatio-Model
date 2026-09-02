import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import io
import requests

st.set_page_config(
    page_title="DSAT — Skin Lesion Segmentation",
    page_icon="🔬",
    layout="wide"
)

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels//reduction,
                      bias=False),
            nn.ReLU(),
            nn.Linear(channels//reduction, channels,
                      bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        b, c, _, _ = x.shape
        avg = self.fc(self.avg_pool(x).view(b, c))
        mx  = self.fc(self.max_pool(x).view(b, c))
        return x * self.sigmoid(avg+mx).view(b, c, 1, 1)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=5):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size,
                              padding=kernel_size//2,
                              bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.max(dim=1, keepdim=True).values
        return x * self.sigmoid(
            self.conv(torch.cat([avg, mx], dim=1))
        )

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16,
                 spatial_kernel=5):
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention(spatial_kernel)
    def forward(self, x):
        return self.spatial(self.channel(x))

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1,
                      bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1,
                      bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.block(x)

class DSATSegmentation(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.backbone = timm.create_model(
            "inception_v3", pretrained=False,
            features_only=True,
            out_indices=(0,1,2,3,4)
        )
        ch = [64, 192, 288, 768, 2048]
        self.cbam = CBAM(ch[-1], reduction=16,
                         spatial_kernel=5)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=ch[-1], nhead=8,
            dim_feedforward=2048,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=2
        )
        dec_chs = [256, 128, 64, 32]
        self.dec4 = ConvBlock(ch[-1]+ch[-2], dec_chs[0])
        self.dec3 = ConvBlock(dec_chs[0]+ch[-3], dec_chs[1])
        self.dec2 = ConvBlock(dec_chs[1]+ch[-4], dec_chs[2])
        self.dec1 = ConvBlock(dec_chs[2]+ch[-5], dec_chs[3])
        self.head = nn.Sequential(
            nn.Conv2d(dec_chs[3], 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_classes, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        s1, s2, s3, s4, s5 = self.backbone(x)
        s5_att = self.cbam(s5)
        b, c, h, w = s5_att.shape
        tokens   = s5_att.flatten(2).permute(0, 2, 1)
        attended = self.transformer(tokens)
        s5_out   = attended.permute(
            0, 2, 1).view(b, c, h, w)
        d4 = F.interpolate(s5_out, size=s4.shape[2:],
                           mode="bilinear",
                           align_corners=False)
        d4 = self.dec4(torch.cat([d4, s4], dim=1))
        d3 = F.interpolate(d4, size=s3.shape[2:],
                           mode="bilinear",
                           align_corners=False)
        d3 = self.dec3(torch.cat([d3, s3], dim=1))
        d2 = F.interpolate(d3, size=s2.shape[2:],
                           mode="bilinear",
                           align_corners=False)
        d2 = self.dec2(torch.cat([d2, s2], dim=1))
        d1 = F.interpolate(d2, size=s1.shape[2:],
                           mode="bilinear",
                           align_corners=False)
        d1 = self.dec1(torch.cat([d1, s1], dim=1))
        out = F.interpolate(d1, size=(224, 224),
                            mode="bilinear",
                            align_corners=False)
        return self.head(out)



@st.cache_resource
def load_model():
    import requests
    import os

    device    = torch.device("cpu")
    ckpt_path = "dsat_soft_label_best.pth"

    # download checkpoint from Hugging Face if not cached
    if not os.path.exists(ckpt_path):
        with st.spinner("Downloading model weights..."):
            url = "https://huggingface.co/emmanueluc322/DSAT-Skin-Lesion-Segmentation/resolve/main/dsat_soft_label_best.zip"
            response = requests.get(url, stream=True)
            total    = int(response.headers.get(
                "content-length", 0
            ))
            progress = st.progress(0)
            downloaded = 0

            with open(ckpt_path, "wb") as f:
                for chunk in response.iter_content(
                    chunk_size=8192
                ):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        progress.progress(
                            min(downloaded / total, 1.0)
                        )
            progress.empty()

    model = DSATSegmentation(num_classes=1)
    model.load_state_dict(
        torch.load(ckpt_path, map_location="cpu")
    )
    model.eval()
    return model, device


# ── inference ──────────────────────────────────────────────
def run_inference(image_pil, model, device):
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    tensor = transform(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(tensor).squeeze().cpu().numpy()

    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = tensor.squeeze().cpu().numpy().transpose(1,2,0)
    img  = np.clip(img * std + mean, 0, 1)

    uncertainty = 1 - np.abs(pred - 0.5) * 2
    binary      = (pred > 0.5).astype(np.float32)

    return img, pred, binary, uncertainty

# ── plot output ────────────────────────────────────────────
def make_figure(img, pred, binary, uncertainty):
    fig = plt.figure(figsize=(16, 4))
    fig.patch.set_facecolor("white")
    gs  = gridspec.GridSpec(1, 4, wspace=0.05)

    panels = [
        (img,         None,          "Input Image"),
        (pred,        "RdYlGn_r",    "Confidence Map"),
        (binary,      "gray",        "Binary Prediction"),
        (uncertainty, "hot",         "Uncertainty Map"),
    ]

    for i, (data, cmap, title) in enumerate(panels):
        ax = fig.add_subplot(gs[i])
        ax.set_facecolor("white")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=11, pad=5,
                     fontweight="bold")
        for sp in ax.spines.values():
            sp.set_edgecolor("#CCCCCC")
        if cmap is None:
            ax.imshow(data)
        else:
            im = ax.imshow(data, cmap=cmap,
                           vmin=0, vmax=1)
            plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150,
                bbox_inches="tight",
                facecolor="white")
    buf.seek(0)
    plt.close()
    return buf

# ── UI ─────────────────────────────────────────────────────
st.title("🔬 DSAT — Skin Lesion Segmentation")
st.markdown("""
**Uncertainty-Aware Dermoscopic Segmentation**
via Dual Attention and Soft Multi-Annotator Supervision

Trained on **IMA++** — 14,967 images · 16 annotators · 17,684 masks
""")

# metrics row
col1, col2, col3, col4 = st.columns(4)
col1.metric("IAA Gap", "−0.0025", "Exceeds human agreement")
col2.metric("ECE", "0.0401", "Well calibrated")
col3.metric("Wilcoxon p", "0.0004", "Statistically significant")
col4.metric("Parameters", "80.8M", "Full architecture")

st.divider()

# upload section
col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("Upload Image")
    uploaded = st.file_uploader(
        "Upload a dermoscopy image",
        type=["jpg", "jpeg", "png"],
        help="Best results with dermoscopic images"
    )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded image",
                 use_column_width=True)

        st.info("""
        **Output guide:**
        - 🔴 Red = high lesion confidence
        - 🟡 Yellow = uncertain boundary
        - 🟢 Green = background
        - ☀️ Bright uncertainty = model unsure
        """)

with col_right:
    if uploaded:
        with st.spinner("Running DSAT segmentation..."):
            model, device = load_model()
            img, pred, binary, uncertainty = \
                run_inference(image, model, device)
            buf = make_figure(
                img, pred, binary, uncertainty
            )

        st.subheader("Segmentation Output")
        st.image(buf, use_column_width=True)

        # download button
        st.download_button(
            label="Download confidence map",
            data=buf,
            file_name="dsat_confidence_map.png",
            mime="image/png"
        )

st.divider()
st.markdown("""
**About DSAT**

DSAT (Dual Attention Self-Attention Transformer) is trained using soft
labels derived from pixel-wise averaging of multiple expert annotator
masks, teaching the model to output intermediate confidence values at
contested boundaries rather than committing to a hard binary decision.

[📄 Paper](#) · [💻 GitHub](https://github.com/EmmanuelUka) ·
[🎓 Kent State University](#)
""")
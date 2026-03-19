# 👗 Virtual Try-On — Built From Scratch

> End-to-end VTON pipeline using **Flow-based Warping** + **Diffusion Models**. No pretrained VTON model. Every component designed and trained from the ground up.

<p align="center">
  <a href="https://huggingface.co/spaces/dmc98/VirtualTryOn_from_scratch">
    <img src="https://img.shields.io/badge/🤗 Live Demo-HuggingFace Spaces-yellow?style=for-the-badge" />
  </a>
  <a href="https://huggingface.co/dmc98/viton_models">
    <img src="https://img.shields.io/badge/📦 Model Weights-HuggingFace-blue?style=for-the-badge" />
  </a>
  <img src="https://img.shields.io/badge/Python-3.10-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/PyTorch-2.0-red?style=for-the-badge" />
</p>

---

## ✨ Results

| Person | Clothing | Try-On Result v1 | Try-On Result v3 |
|--------|----------|------------------|------------------|
| <img width="150" src="https://github.com/user-attachments/assets/0ae36f07-c350-46a4-8fb4-0d124d1f4311" /> | <img width="150" src="https://github.com/user-attachments/assets/481bdc94-f756-451a-ab11-95b0c2d6a92c" /> | <img width="150" src="https://github.com/user-attachments/assets/a8aa9120-7f72-43e6-aa6a-12adefed5824" /> |<img width="200" height="256" alt="image" src="https://github.com/user-attachments/assets/d2defbe8-3beb-419c-b7b9-0092215c00e1" />
| <img width="150" src="https://github.com/user-attachments/assets/0ae36f07-c350-46a4-8fb4-0d124d1f4311" /> | <img width="150" src="https://github.com/user-attachments/assets/ec42dfd8-f095-4e70-a1ff-281c3fb258e0" /> | <img width="150" src="https://github.com/user-attachments/assets/490147d0-92a1-4a2e-9d0d-7b578ce4b91e" /> |<img width="200" height="256" alt="image" src="https://github.com/user-attachments/assets/75188fc7-8e93-46b6-a388-12049a0a7d16" />
| <img width="150" src="https://github.com/user-attachments/assets/852d98fe-fd0d-495e-91ad-1eb3f7a91524" /> | <img width="150" src="https://github.com/user-attachments/assets/4464a890-aafc-4765-a480-1e9a5c75ac86" /> | <img width="150" src="https://github.com/user-attachments/assets/c61a7bfc-8562-4d24-8fa6-144fff226a16" /> |<img width="200" height="256" alt="image" src="https://github.com/user-attachments/assets/2d2968fe-71a3-4138-8fd0-f1718d9acf1b" />

> 🔝 Last column = latest model (self-attention U-Net, 5k dataset, batch size 10)

---

## 🚀 Try It Live

**No setup needed** → [huggingface.co/spaces/dmc98/VirtualTryOn_from_scratch](https://huggingface.co/spaces/dmc98/VirtualTryOn_from_scratch)

Upload a person image + a clothing image → get the try-on result in seconds.

---

## 🧠 How It Works

The pipeline has three stages:

```
Person Image + Clothing Image
        │
        ▼
┌─────────────────────┐
│   1. Preprocessing  │  DensePose · Body Segmentation · Agnostic Mask
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  2. Warping Network │  Geometrically aligns clothing to body shape
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  3. Diffusion Model │  Synthesizes the final photorealistic try-on
└──────────┬──────────┘
           │
           ▼
     ✅ Try-On Result
```

### Stage 1 — Preprocessing (`parsehuman.py`)

From the **person image**:
- DensePose map (Detectron2)
- Body segmentation (SegFormer-B2)
- Keypoints (Detectron2)
- Agnostic image + mask (clothing region removed)

From the **clothing image**:
- Cloth mask (rembg)

### Stage 2 — Flow-Based Warping Network

Multi-scale flow prediction with two parallel branches:
- **Cloth branch** → cloth image + cloth mask
- **Body branch** → DensePose + body segmentation

Output: warped cloth geometrically aligned to the person's body.

📄 Architecture based on: [arxiv.org/pdf/2206.14180](https://arxiv.org/pdf/2206.14180)

### Stage 3 — Diffusion Model

- **VAE encoder**: 512×512 images → 64×64 latent tensors (f=8)
- **Noise scheduler**: cosine betas
- **U-Net**: Stable Diffusion-style backbone with self-attention + SE blocks + Adaptive Normalization
- **Dual-input design**: separate paths for original and agnostic latents
- **Training losses**: MSE (noise prediction) + Perceptual/VGG19
- **Inference**: DDIM sampler, 200 steps

📄 Inspired by: [arxiv.org/pdf/2308.06101](https://arxiv.org/pdf/2308.06101)

---

## 📈 Training Progress

| Version | Dataset | Batch Size | U-Net | Notable Change |
|---------|---------|------------|-------|----------------|
| v1 | 2k images | 1 | Basic | Initial pipeline |
| v2 | 3k images | 1 | Basic | Warping improvements |
| v3 | 5k images | 10 | SD-style + Self-Attention | **Current best results** |

---

## ⚙️ Quickstart

```bash
git clone https://github.com/DanMenCas/VirtualTryOn_from_scratch
cd VirtualTryOn_from_scratch
pip install -r requirements.txt
python app.py
```

Download model weights from [HuggingFace](https://huggingface.co/dmc98/viton_models) and place in the root directory.

---

## 🗂️ Project Structure

```
├── parsehuman.py          # Full preprocessing pipeline
├── warpingnetwork.py      # Flow-based warping model
├── diffusionmodel.py      # U-Net + VAE + noise scheduler
├── losses.py              # MSE + perceptual loss
├── viton_pipeline.py      # End-to-end inference pipeline
├── app.py                 # Gradio demo app
├── Train_WarpingNetwork.ipynb
└── Train_DiffusionModel.ipynb
```

---

## 🔭 Roadmap

- [ ] Cross-attention for text-guided try-on
- [ ] Dataset scale to 10k+ images
- [ ] REST API for e-commerce integration
- [ ] Multi-garment support (pants, dresses)

---

## 📚 References

- [VITON-HD Dataset](https://github.com/shadow2496/VITON-HD)
- [HR-VITON — Warping architecture](https://arxiv.org/pdf/2206.14180)
- [DCI-VTON — Diffusion architecture](https://arxiv.org/pdf/2308.06101)

---

## 👤 Author

**Daniel Mendoza** · ML Engineer  
[resviss.com](https://www.resviss.com) · [HuggingFace](https://huggingface.co/dmc98)

> Built with limited compute to prove the pipeline works. Imagine what's possible at scale.

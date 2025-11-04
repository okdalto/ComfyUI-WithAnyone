# ComfyUI-WithAnyone

![Preview](https://github.com/okdalto/ComfyUI-WithAnyone/raw/main/assets/preview.jpg)

This repository is a **ComfyUI port** of the **WithAnyone** model introduced in the paper  
> [WithAnyone: Towards Controllable and ID-Consistent Image Generation (2025)](https://arxiv.org/abs/2510.14975)

Original implementation: [Doby-Xu/WithAnyone](https://github.com/Doby-Xu/WithAnyone).  
Huge thanks and congratulations to the authors for their excellent work and for releasing it as open source.

---

## 🧩 Installation

Clone this repository under your ComfyUI custom nodes directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/okdalto/ComfyUI-WithAnyone
cd ComfyUI-WithAnyone
pip install -r requirements.txt
```

---

## 📦 Model Setup

### 1. Text Encoder Models

Go to `ComfyUI/models/clip` and download the following:

```bash
wget https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors
wget https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors
```

### 2. Diffusion Models

Go to `ComfyUI/models/diffusion_models` and download:

```bash
wget https://huggingface.co/bstungnguyen/Flux/resolve/main/flux1-dev.safetensors
wget https://huggingface.co/WithAnyone/WithAnyone/resolve/main/withanyone.safetensors
```

### 3. VAE Model

Go to `ComfyUI/models/vae` and download:

```bash
wget https://huggingface.co/bstungnguyen/Flux/resolve/main/vae/diffusion_pytorch_model.safetensors
```

### 4. SigLIP Model

Go to `ComfyUI/models/diffusers` and download the multilingual SigLIP model:

```bash
apt update
apt install git-lfs -y
git lfs install
git clone https://huggingface.co/google/siglip-base-patch16-256-multilingual
```

---

## ⚠️ Important Note

The `WithAnyone Model Loader` will automatically download the **ArcFace** model into  
`ComfyUI/custom_nodes/ComfyUI-WithAnyone/models/`.

However, there is a known issue. If you encounter an error such as:

```
assert 'detection' in self.models
```

please manually move the downloaded model directory to the correct location under the path above.

---

## ⚙️ Usage

1. Load workflow by the given `workflow.png` or `workflow_demo.json` files.
2. Upload 1~4 reference images, each containing a single face.
3. [Recommended] Use bbox calculation nodes to determine where to place the faces in the output image.
4. [Recommended] Adjust `siglip_weight` in the Sampler node. Higher values yield better identity consistency but may reduce adherence to the text prompt.


---

## 🚧 TODO

- **LoRA support** are planned for future updates.
- **Flux Kontext** model support has not yet been tested.


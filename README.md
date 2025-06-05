<h1 align="center">PnP-Nystra : Plug-and-Play Linear Attention for Pre-Trained Image and Video Restoration Models</h1>

<div align="center">

  [![Status](https://img.shields.io/badge/status-active-success.svg)]() 
  [![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

</div>

---
<p align="center">
  PnP-Nystra is a Nyström-based, <strong>training-free</strong> replacement for MHSA in image/video restoration models (e.g., SwinIR, Uformer, RVRT). It delivers <strong>2–4× GPU</strong> and <strong>2–5× CPU</strong> inference speedups with under 1.5 dB PSNR loss on denoising, deblurring, and super-resolution tasks.
  <br>
</p>


## 📝 Table of Contents
- [About](#about)
- [Prerequisites](#getting_started)
- [Running inference](#tests)
- [Timing analysis](#timing)
- [Authors](#authors)
- [Acknowledgments](#acknowledgement)

## About <a name = "about"></a>
Write about 1-2 paragraphs describing the purpose of your project.

## Prerequisites <a name = "getting_started"></a>

1. **Clone the repository**

   ```bash
   git clone https://github.com/Srinivas-512/PnP_Nystra
   cd PnP_Nystra
   ```

2. **Download pretrained models**

   * Go to the Google Drive link for pretrained models:

     ```
     https://drive.google.com/drive/folders/1G0jb_yN6aYotbcPNhWj_3JdAOjzgLFb8?usp=sharing
     ```
   * Download the weight folders for the models and extract their contents.
   * Place everything under:

     ```
     ./pretrained_models/
     ```

     After this step you should have the following structure:

     ```
     PnP_Nystra/
       └─ pretrined_models/
           ├─ RVRT/
           └─ SwinIR/
           └─ Uformer/
     ```

3. **Download datasets**

   * Go to the Google Drive link for datasets:

     ```
     https://drive.google.com/drive/folders/1abRCvUDrrRYnxjzhzkdbZIgJAKabGv7T?usp=sharing
     ```
   * Download the dataset folders for the models and extract their contents.
   * Place the dataset files under:

     ```
     ./datasets/
     ```

     After this step you should have the following structure:

     ```
     PnP_Nystra/
       └─ datasets/
           ├─ RVRT/
           ├─ SwinIR/
           └─ Uformer/
     ```

4. **Installing dependencies**
   Make sure you have Python ≥3.7 installed. From within the repository root, run:  
   ```bash
   pip install -r requirements.txt
   ```



## Running inference <a name = "tests"></a>

Below is a concise summary of supported datasets/models, followed by short subsections for each model showing only the flags relevant to experimentation as presented in the paper (`--folder_*` or `--input_dir`/`--result_dir`, `--model_path`, `--mech`, `--device`). All example commands use PnP-Nystra as the proposed method, but this can be switched out for original Window Attention as well.

---

|    Model    | Task                        | Dataset Options                                | Model File(s)                |
| :---------: | :-------------------------- | :--------------------------------------------- | :--------------------------- |
|  **SwinIR** | Super-Resolution (×2,×4,×8) | Set5, BSDS100                                  | `x2.pth`, `x4.pth`, `x8.pth` |
| **Uformer** | Denoising / Deblurring      | SIDD, BSDS200 (denoise)<br>RealBlur-R (deblur) | `denoise.pth`, `deblur.pth`  |
|   **RVRT**  | Video Restoration           | REDS4, Vid4                                    | `REDS.pth`, `Vid.pth`        |

> **Note:**
>
> * All pre-trained weights live under `pretrained_models/<ModelName>/…`.
> * All datasets are stored under `datasets/<ModelName>/…`.
> * This follows from steps 2 and 3 of Prerequisites above

---

### 1. SwinIR (Super-Resolution)

```bash
python run_swinir.py \
    --scale 8 \
    --model_path pretrained_models/SwinIR/x8.pth \
    --folder_lq  datasets/SwinIR/Set5/LR_bicubic/X8 \
    --folder_gt  datasets/SwinIR/Set5/HR \
    --mech       pnp_nystra \
    --device     cuda
```

* `--scale`: upscaling factor (2, 4 or 8).
* `--model_path`: path to the `.pth` file (e.g. `x8.pth`).
* `--folder_lq` / `--folder_gt`: low-/high-resolution folders.
* `--mech`: set to `pnp_nystra` for the proposed method / `original` for original window attention.
* `--device`: set to `cuda` for GPU / `cpu` for CPU execution.

---

### 2. Uformer

#### 2.1 Denoising (SIDD or BSDS200)

```bash
python run_uformer.py \
    --input_dir   datasets/Uformer/SIDD/val \
    --result_dir  ./results/denoising/SIDD/ \
    --weights     pretrained_models/Uformer/denoise.pth \
    --mech        pnp_nystra \
    --device      cuda
```

* `--input_dir`: folder of noisy images (e.g. `datasets/Uformer/SIDD/val`).
* `--result_dir`: folder where outputs will be saved.
* `--weights`: path to `denoise.pth`.
* `--mech`: set to `pnp_nystra` for the proposed method / `original` for original window attention.
* `--device`: set to `cuda` for GPU / `cpu` for CPU execution.

#### 2.2 Deblurring (RealBlur-R)

```bash
python run_uformer.py \
    --input_dir   datasets/Uformer/RealBlur-R/val \
    --result_dir  ./results/deblurring/RealBlur-R/ \
    --weights     pretrained_models/Uformer/deblur.pth \
    --mech        pnp_nystra \
    --device      cuda
```

* `--input_dir`: folder of blurred images (e.g. `datasets/Uformer/RealBlur-R/val`).
* `--result_dir`: folder where outputs will be saved.
* `--weights`: path to `deblur.pth`.
* `--mech`: set to `pnp_nystra` for the proposed method / `original` for original window attention.
* `--device`: set to `cuda` for GPU / `cpu` for CPU execution.

---

### 3. RVRT (Video Restoration)

```bash
python run_rvrt.py \
    --folder_lq       datasets/RVRT/Vid4/BDx4 \
    --folder_gt       datasets/RVRT/Vid4/GT \
    --model_path      pretrained_models/RVRT/Vid.pth \
    --tile            10 64 64 \
    --tile_overlap    2 20 20 \
    --mech            pnp_nystra
```

* `--folder_lq` / `--folder_gt`: low-quality and ground-truth video folders.
* `--model_path`: path to `Vid.pth` (for Vid4) or `REDS.pth` (for REDS4).
* `--tile` / `--tile_overlap`: how the input is tiled (e.g. `10 64 64` and `2 20 20` to reproduce paper experiments).
* `--mech`: set to `pnp_nystra` for the proposed method / `original` for original window attention.

---

You can swap `<Dataset>`, `<scale>`, `<model>.pth`, `device` and `mech` as needed to reproduce experiments from the paper.


## Timing analysis <a name="timing"></a>
Add notes about how to use the system.

## Authors <a name = "authors"></a>
- [Srinivasan Kidambi](https://github.com/Srinivas-512)
- [Dr. Pravin Nair](https://github.com/pravin1390)


## Acknowledgements <a name = "acknowledgement"></a>
- Hat tip to anyone whose code was used
- Inspiration
- References

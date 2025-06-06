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
- [Prerequisites](#getting_started)
- [Running inference](#tests)
- [Timing analysis](#timing)
- [Authors](#authors)
- [Acknowledgments](#acknowledgement)


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
   This code is tested on Python 3.12.2. From within the repository root, run:  
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

### 1. SwinIR (Image Super-Resolution)

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

#### 2.1 Image Denoising (SIDD or BSDS200)

```bash
python run_uformer.py \
    --input_dir   datasets/Uformer/SIDD/val \
    --result_dir  ./results/denoising/SIDD/ \
    --weights     pretrained_models/Uformer/denoise.pth \
    --dataset     SIDD \
    --mech        pnp_nystra \
    --device      cuda
```

* `--input_dir`: folder of noisy images (e.g. `datasets/Uformer/SIDD/val`).
* `--result_dir`: folder where outputs will be saved.
* `--weights`: path to `denoise.pth`.
* `--dataset`: dataset being tested (SIDD/BSDS)
* `--mech`: set to `pnp_nystra` for the proposed method / `original` for original window attention.
* `--device`: set to `cuda` for GPU / `cpu` for CPU execution.

#### 2.2 Image Deblurring (RealBlur-R)

```bash
python run_uformer.py \
    --input_dir   datasets/Uformer/RealBlur-R/val \
    --result_dir  ./results/deblurring/RealBlur-R/ \
    --weights     pretrained_models/Uformer/deblur.pth \
    --dataset     RealBlur_R
    --mech        pnp_nystra \
    --device      cuda
```

* `--input_dir`: folder of blurred images (e.g. `datasets/Uformer/RealBlur-R/val`).
* `--result_dir`: folder where outputs will be saved.
* `--weights`: path to `deblur.pth`.
* `--mech`: set to `pnp_nystra` for the proposed method / `original` for original window attention.
* `--device`: set to `cuda` for GPU / `cpu` for CPU execution.

---

### 3. RVRT (Video Super-Resolution)

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

The file `timing_analysis.py` provides a simple script for measuring and comparing forward‐pass times between the original window‐attention mechanism and the proposed PnP-Nystra attention. This is used solely for the ablation study on effect of sequence length on inference time in the paper. **This timing script is *not* used during inference for any individual model.**

**How to use:**

1. Open `timing_analysis.py` and modify the following variables in the `__main__` block as needed:

   * `all_sizes`: a list of window‐side lengths (e.g., `[16, 32, 64, 128]`). Each entry defines a square window of size `window_size × window_size` (so sequence length = `window_size²`).
   * `num_landmarks`: number of landmarks (default `16`).
   * `iters`: number of iterations for the Moore-Penrose pseudo-inverse (default `3`).
   * `device`: either `'cpu'` or `'cuda'` (default `'cpu'`).

2. Run the script as-is. For example:

   ```bash
   python timing_analysis.py
   ```

   * The script will iterate over each `window_size` in `all_sizes`, compute MSE between original and PnP-Nystra attention outputs (for sanity check), then measure average runtime (in ms) over 100 repeats (discarding the first 5 iterations as warmup) for both original and PnP-Nystra attention.

3. Inspect the printed outputs. For each window size, you’ll see:

   * The squared‐window size (e.g., “Testing window size 16 with 256 tokens”).
   * The MSE error between attention outputs (should be very small).
   * “PnP-Nystra takes (ms)” → average forward time for PnP-Nystra.
   * “Original takes (ms)” → average forward time for the default softmax attention.

Because this is meant for the ablation study (to demonstrate how runtime scales with sequence length), there is no additional command‐line interface—everything is configured by editing the script’s `__main__` section. You can change `all_sizes`, `num_landmarks`, `iters`, and `device` to reproduce or extend the timing analysis reported in the ablation study.


## Authors <a name = "authors"></a>
- [Srinivasan Kidambi](https://github.com/Srinivas-512) 
- [Dr. Pravin Nair](https://github.com/pravin1390) (Dept. of Electrical Engineering, IIT Madras)


## Acknowledgements <a name = "acknowledgement"></a>
This repository builds upon and integrates the original codebases of:

* [SwinIR](https://github.com/JingyunLiang/SwinIR) by Jingyun Liang et al.
* [Uformer](https://github.com/ZhendongWang6/Uformer) by Zhendong Wang et al.
* [RVRT](https://github.com/JingyunLiang/RVRT) by Jingyun Liang et al.

We gratefully acknowledge the authors for their excellent work and for releasing their implementations under open licenses.

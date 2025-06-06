import os
import sys
import argparse
import glob
import math

import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader
from PIL import Image
from skimage import img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss
from tqdm import tqdm
import scipy.io as sio

# === Add project root and dataset paths to PYTHONPATH ===
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name, "../dataset/"))
sys.path.append(os.path.join(dir_name, ".."))

# === Import model & utility loaders ===
from models.uformer import Uformer
from utils.uformer_utils import load_checkpoint
# BSDS/SIDD denoising loader
from utils.uformer_denoise_dataset_utils import get_test_data as get_bsds_test_data
# RealBlur deblurring loader
from utils.uformer_deblur_dataset_utils import get_test_data as get_realblur_test_data


def mkdir(path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def save_img(filepath, img_rgb):
    """Save a NumPy image (H×W×3, RGB float [0,1]) to disk as PNG."""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filepath, img_bgr)


def expand2square(timg: torch.Tensor, factor: float = 16.0):
    """
    Pad a tensor to a square whose side is the smallest multiple of `factor`
    ≥ max(H, W). Returns (padded_image, mask) where mask==1 inside original region.
    """
    _, _, h, w = timg.size()
    X = int(math.ceil(max(h, w) / float(factor)) * factor)

    img_padded = torch.zeros(1, 3, X, X).type_as(timg)
    mask = torch.zeros(1, 1, X, X).type_as(timg)

    h_start = (X - h) // 2
    w_start = (X - w) // 2
    img_padded[:, :, h_start : h_start + h, w_start : w_start + w] = timg
    mask[:, :, h_start : h_start + h, w_start : w_start + w].fill_(1)
    return img_padded, mask


def read_image_pil(image_path: str):
    """Read an image via PIL and return as NumPy array (HxWx3, uint8)."""
    return np.array(Image.open(image_path))


def evaluate_metrics(restored_dir: str, original_dir: str):
    """
    Given two folders of PNGs—`restored_dir` and `original_dir`—compute PSNR/SSIM
    for every filename that appears in both. If `verbose=True`, print each image's
    scores; always print the overall average at the end.

    Parameters:
        restored_dir: path to folder containing restored .png
        original_dir: path to folder containing ground-truth .png
        verbose: if True, prints per-file PSNR/SSIM
    """
    restored_files = glob.glob(os.path.join(restored_dir, "*.png"))
    original_files = glob.glob(os.path.join(original_dir, "*.png"))

    restored_names = {os.path.basename(f) for f in restored_files}
    original_names = {os.path.basename(f) for f in original_files}
    common_names = sorted(restored_names & original_names)

    if not common_names:
        print(f"No matching filenames between:\n  {restored_dir}\n  {original_dir}")
        return

    psnr_vals, ssim_vals = [], []

    for fname in common_names:
        im_r = np.array(Image.open(os.path.join(restored_dir, fname)))
        im_o = np.array(Image.open(os.path.join(original_dir, fname)))

        val_ssim = ssim_loss(im_o, im_r, channel_axis=2, data_range=255)
        val_psnr = psnr_loss(im_o, im_r, data_range=255)

        psnr_vals.append(val_psnr)
        ssim_vals.append(val_ssim)


    avg_ssim = np.mean(ssim_vals)
    avg_psnr = np.mean(psnr_vals)
    print(f"→ Average PSNR: {avg_psnr:.2f}, Average SSIM: {avg_ssim:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Unified test script for BSDS (denoising), RealBlur (deblurring), and SIDD (denoising)."
    )
    # Common arguments
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["BSDS", "RealBlur_R", "SIDD"],
        help="Which dataset to evaluate: 'BSDS', 'RealBlur_R', or 'SIDD'",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Root input directory (for BSDS this is the BSDS200 folder, for RealBlur it's the parent of RealBlur_R, for SIDD it's the folder containing ValidationNoisyBlocksSrgb.mat).",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        required=True,
        help="Directory where output images (and mats for SIDD) will be saved.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to the trained Uformer .pth checkpoint.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="CUDA_VISIBLE_DEVICES (e.g. '0,1').",
    )
    parser.add_argument(
        "--arch", type=str, default="Uformer_B", help="Model architecture name (appears in result paths)."
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for DataLoader (if used)."
    )
    parser.add_argument(
        "--save_images",
        action="store_true",
        help="If set, save restored RGB images in result_dir/png/. (Ignored for SIDD; SIDD always saves PNG).",
    )
    parser.add_argument(
        "--embed_dim", type=int, default=32, help="Uformer embed_dim (default=32)."
    )
    parser.add_argument(
        "--win_size", type=int, default=8, help="Attention window size (base value; may be overridden per dataset)."
    )
    parser.add_argument(
        "--token_projection",
        type=str,
        default="linear",
        help="linear/conv token projection.",
    )
    parser.add_argument(
        "--token_mlp",
        type=str,
        default="leff",
        help="ffn/leff token MLP.",
    )
    parser.add_argument(
        "--dd_in", type=int, default=3, help="Number of input channels (3 for RGB)."
    )
    parser.add_argument(
        "--mech",
        type=str,
        default="original",
        choices=["original", "pnp_nystra"],
        help="Attention mechanism: 'original' or 'pnp_nystra'.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (e.g. 'cuda')."
    )

    # ViT-specific args (unused here but kept for compatibility)
    parser.add_argument("--vit_dim", type=int, default=256, help="ViT hidden dim.")
    parser.add_argument("--vit_depth", type=int, default=12, help="ViT depth.")
    parser.add_argument("--vit_nheads", type=int, default=8, help="ViT number of heads.")
    parser.add_argument("--vit_mlp_dim", type=int, default=512, help="ViT MLP dim.")
    parser.add_argument("--vit_patch_size", type=int, default=16, help="ViT patch size.")
    parser.add_argument(
        "--global_skip",
        action="store_true",
        default=False,
        help="Use global skip connection in Uformer.",
    )
    parser.add_argument(
        "--local_skip",
        action="store_true",
        default=False,
        help="Use local skip connection in Uformer.",
    )
    parser.add_argument(
        "--vit_share",
        action="store_true",
        default=False,
        help="Share ViT module across Uformer stages.",
    )
    parser.add_argument(
        "--train_ps", type=int, default=128, help="(Unused) patch size."
    )

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    device = torch.device(args.device)

    # Create top-level result directories
    mkdir(args.result_dir)
    result_png_dir = os.path.join(args.result_dir, "png")
    mkdir(result_png_dir)
    if args.dataset == "SIDD":
        # SIDD also uses a 'mat' directory
        result_mat_dir = os.path.join(args.result_dir, "mat")
        mkdir(result_mat_dir)

    # Common Uformer depths
    DEPTHS = [1, 2, 8, 8, 2, 8, 8, 2, 1]

    # === Dataset-specific logic ===

    if args.dataset == "BSDS":
        #
        # 1) BSDS denoising evaluation
        #
        WINDOW_SIZES = [64]
        test_dataset = get_bsds_test_data(os.path.join(args.input_dir, "input"))
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=8,
            drop_last=False,
        )

        for win_size in WINDOW_SIZES:
            if args.mech == "original":
                num_landmarks, iters = 16, 1
            else:  # pnp_nystra
                num_landmarks, iters = 32, 6

            model = Uformer(
                attention_mechanism=args.mech,
                num_landmarks=num_landmarks,
                iters=iters,
                img_size=128,
                embed_dim=args.embed_dim,
                win_size=win_size,
                token_projection=args.token_projection,
                token_mlp=args.token_mlp,
                depths=DEPTHS,
                modulator=True,
                dd_in=args.dd_in,
            )
            load_checkpoint(model, args.weights)
            model.to(device)
            model.eval()

            with torch.no_grad():
                for _, data in enumerate(tqdm(test_loader), 0):
                    noisy, filenames = data[0].to(device), data[1]
                    _, _, h, w = noisy.shape
                    noisy_padded, mask = expand2square(noisy, factor=128)
                    restored = model(noisy_padded)
                    restored = (
                        torch.masked_select(restored, mask.bool())
                        .reshape(1, 3, h, w)
                        .clamp(0, 1)
                        .cpu()
                        .numpy()
                        .squeeze()
                        .transpose(1, 2, 0)
                    )
                    out_png = img_as_ubyte(restored)
                    save_path = os.path.join(result_png_dir, f"{filenames[0]}.png")
                    save_img(save_path, out_png)

        # Evaluate metrics (BSDS ground-truth in BSDS200/target)
        bsds_gt_dir = os.path.join(args.input_dir, "target")
        print("BSDS metrics:")
        evaluate_metrics(result_png_dir, bsds_gt_dir)

    elif args.dataset == "RealBlur_R":
        #
        # 2) RealBlur motion-deblurring evaluation
        #
        WINDOW_SIZES = [32]
        for win_size in WINDOW_SIZES:
            for ds_name in args.dataset.split(","):
                rgb_dir_test = os.path.join(args.input_dir, ds_name, "test", "input")
                test_dataset = get_realblur_test_data(rgb_dir_test, img_options={})
                test_loader = DataLoader(
                    dataset=test_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=4,
                    drop_last=False,
                    pin_memory=False,
                )

                result_dir_ds = os.path.join(args.result_dir, ds_name, args.arch)
                mkdir(result_dir_ds)

                if args.mech == "original":
                    num_landmarks, iters = 16, 1
                else:  # pnp_nystra
                    num_landmarks, iters = 16, 3

                model = Uformer(
                    attention_mechanism=args.mech,
                    num_landmarks=num_landmarks,
                    iters=iters,
                    img_size=128,
                    embed_dim=args.embed_dim,
                    win_size=win_size,
                    token_projection=args.token_projection,
                    token_mlp=args.token_mlp,
                    depths=DEPTHS,
                    modulator=True,
                    dd_in=args.dd_in,
                )
                load_checkpoint(model, args.weights)
                model.to(device)
                model.eval()

                with torch.no_grad():
                    for _, data in enumerate(tqdm(test_loader), 0):
                        torch.cuda.empty_cache()
                        inp, filenames = data[0].to(device), data[1]
                        _, _, h, w = inp.shape
                        inp_padded, mask = expand2square(inp, factor=128)
                        restored = model(inp_padded)
                        restored = (
                            torch.masked_select(restored, mask.bool())
                            .reshape(1, 3, h, w)
                            .clamp(0, 1)
                            .cpu()
                            .numpy()
                            .squeeze()
                            .transpose(1, 2, 0)
                        )
                        out_png = img_as_ubyte(restored)
                        save_path = os.path.join(result_dir_ds, f"{filenames[0]}.png")
                        save_img(save_path, out_png)

                # Evaluate metrics (RealBlur ground-truth in <ds_name>/<ds_name>/test/target)
                realblur_gt_dir = os.path.join(args.input_dir, ds_name, "test", "target")
                print(f"{ds_name} metrics:")
                evaluate_metrics(result_dir_ds, realblur_gt_dir)

    elif args.dataset == "SIDD":
        #
        # 3) SIDD denoising evaluation (ValidationNoisyBlocksSrgb.mat)
        #
        WINDOW_SIZES = [64]
        mat_path = args.input_dir + "/ValidationNoisyBlocksSrgb.mat"
        mat_data = sio.loadmat(mat_path)
        Inoisy = (mat_data["ValidationNoisyBlocksSrgb"] / 255.0).astype(np.float32)
        num_images, num_patches, H, W, _ = Inoisy.shape
        print(f"SIDD noisy blocks shape: {Inoisy.shape}")

        restored_all = np.zeros_like(Inoisy)

        for win_size in WINDOW_SIZES:
            if args.mech == "original":
                num_landmarks, iters = 16, 1
            else:
                num_landmarks, iters = 32, 6

            model = Uformer(
                attention_mechanism=args.mech,
                num_landmarks=num_landmarks,
                iters=iters,
                img_size=H,
                embed_dim=args.embed_dim,
                win_size=win_size,
                token_projection=args.token_projection,
                token_mlp=args.token_mlp,
                depths=DEPTHS,
                modulator=True,
                dd_in=args.dd_in,
            )
            load_checkpoint(model, args.weights)
            model.to(device)
            model.eval()

            with torch.no_grad():
                for i in range(num_images):
                    for k in range(num_patches):
                        noisy_patch = (
                            torch.from_numpy(Inoisy[i, k, :, :, :])
                            .unsqueeze(0)
                            .permute(0, 3, 1, 2)
                            .to(device)
                        )  # 1×3×H×W
                        _, _, h, w = noisy_patch.shape
                        patch_padded, mask = expand2square(noisy_patch, factor=128)
                        restored_patch = model(patch_padded)
                        restored_patch = (
                            torch.masked_select(restored_patch, mask.bool())
                            .reshape(1, 3, h, w)
                            .clamp(0, 1)
                            .cpu()
                            .permute(0, 2, 3, 1)
                            .squeeze(0)
                            .numpy()
                        )
                        restored_all[i, k, :, :, :] = restored_patch

                        out_png = img_as_ubyte(restored_patch)
                        fname = f"{i+1:04d}_{k+1:02d}.png"
                        save_path = os.path.join(result_png_dir, fname)
                        save_img(save_path, out_png)

            mat_out_path = os.path.join(args.result_dir, "mat", "SIDD_restored.mat")
            sio.savemat(mat_out_path, {"RestoredBlocksSrgb": restored_all})

        # Evaluate metrics (SIDD ground-truth in SIDD/val/groundtruth)
        sidd_gt_dir = os.path.join(args.input_dir, "groundtruth")
        print("SIDD metrics:")
        evaluate_metrics(result_png_dir, sidd_gt_dir)

    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")


if __name__ == "__main__":
    main()

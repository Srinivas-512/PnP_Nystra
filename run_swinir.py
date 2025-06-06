import argparse
import cv2
import glob
import numpy as np
from collections import OrderedDict
import os
import torch
import random
random.seed(42)


from models.swinir import SwinIR as net
from utils import swinir_utils as util

def process_and_evaluate(
    args,
    window_size,
    mech,
    device,
    num_landmarks=None,
    iters=None,
    sample_k=None,
):
    """
    Common routine for both 'original' and 'new' mechanisms.
    - mech: 'original' or 'pnp_nystra'.
    - For mech == 'original', num_landmarks/iters are ignored (we hardcode 16,1).W
    - sample_k: if not None, randomly sample `sample_k` image paths from the folder.
    """

    # 1) Setup folder, save_dir, border, and make sure the directory exists
    folder, save_dir, border, window_size = setup(args, window_size)
    os.makedirs(save_dir, exist_ok=True)

    # 2) Prepare dicts to accumulate PSNR/SSIM values
    test_results = OrderedDict([
        ("psnr", []),
        ("ssim", []),
    ])

    # 3) Collect all image paths
    all_paths = sorted(glob.glob(os.path.join(folder, "*")))

    if sample_k is not None:
        # randomly pick `sample_k` images (without replacement)
        sampled = random.sample(all_paths, sample_k)
        paths_to_process = sorted(sampled)
    else:
        paths_to_process = all_paths

    # 4) Iterate over each selected image
    for path in paths_to_process:
        # 4.1) Read LQ + GT
        imgname, img_lq, img_gt = get_image_pair(args, path)  # HWC‐BGR, float32

        # 4.2) Convert LQ to CHW‐RGB tensor
        if img_lq.ndim == 3 and img_lq.shape[2] == 3:
            # BGR → RGB
            img_lq = img_lq[:, :, [2, 1, 0]]
        img_lq = np.transpose(img_lq, (2, 0, 1))              # HWC → CHW
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)  # 1×C×H×W

        # 4.3) Pad so H & W are multiples of window_size
        _, _, h_old, w_old = img_lq.size()
        h_pad = (h_old // window_size + 1) * window_size - h_old
        w_pad = (w_old // window_size + 1) * window_size - w_old

        # vertical flip pad, then trim
        img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], dim=2)[:, :, : h_old + h_pad, :]
        # horizontal flip pad, then trim
        img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], dim=3)[:, :, :, : w_old + w_pad]

        # 4.4) Build model according to mech
        img_size = img_lq.size(-1)  # after padding, height == width
        if mech == "original":
            # hardcode num_landmarks=16, iters=1 (garbage values)
            model = define_model(args, mech, 16, 1, window_size, img_size)
        else:
            # 'pnp_nystra' mech: use the passed-in num_landmarks & iters
            model = define_model(args, mech, num_landmarks, iters, window_size, img_size)

        model.eval()
        model = model.to(device)

        # 4.5) Inference
        with torch.no_grad():
            output = test(img_lq, model, args, window_size)[0]  # we only need the first returned value

        # 4.6) Crop model output back to (h_old * scale, w_old * scale)
        output = output[..., : h_old * args.scale, : w_old * args.scale]

        # 4.7) Convert to NumPy uint8 BGR and save
        output_np = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output_np.ndim == 3:
            # CHW‐RGB → HWC‐BGR
            output_np = np.transpose(output_np[[2, 1, 0], :, :], (1, 2, 0))
        output_np = (output_np * 255.0).round().astype(np.uint8)
        cv2.imwrite(f"{save_dir}/{imgname}_SwinIR.png", output_np)

        # 4.8) If GT exists, compute PSNR/SSIM
        if img_gt is not None:
            img_gt_np = (img_gt * 255.0).round().astype(np.uint8)
            img_gt_np = img_gt_np[: h_old * args.scale, : w_old * args.scale, ...]
            img_gt_np = np.squeeze(img_gt_np)

            psnr_val = util.calculate_psnr(output_np, img_gt_np, crop_border=border)
            ssim_val = util.calculate_ssim(output_np, img_gt_np, crop_border=border)
            test_results["psnr"].append(psnr_val)
            test_results["ssim"].append(ssim_val)

    # 5) Summarize PSNR/SSIM over all processed images
    if len(test_results["psnr"]) > 0:
        ave_psnr = sum(test_results["psnr"]) / len(test_results["psnr"])
        ave_ssim = sum(test_results["ssim"]) / len(test_results["ssim"])
    else:
        ave_psnr = ave_ssim = None

    print(f"{ave_psnr:.2f}, {ave_ssim:.4f}")

    return test_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', type=int, default=1, help='scale factor: 1, 2, 3, 4, 8') # 1 for dn and jpeg car
    parser.add_argument('--noise', type=int, default=15, help='noise level: 15, 25, 50')
    parser.add_argument('--jpeg', type=int, default=40, help='scale factor: 10, 20, 30, 40')
    parser.add_argument('--model_path', type=str,
                        default='model_zoo/swinir/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth')
    parser.add_argument('--folder_lq', type=str, default=None, help='input low-quality test image folder')
    parser.add_argument('--folder_gt', type=str, default=None, help='input ground-truth test image folder')
    parser.add_argument('--tile', type=int, default=None, help='Tile size, None for no tile during testing (testing as a whole)')
    parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
    parser.add_argument('--mech', type = str, default = 'original')
    parser.add_argument('--sample_k', type=int, default=None, help='If not None, randomly sample `sample_k` images from the folder')
    parser.add_argument('--device', type = str, default = 'cuda', help='Device to run the model on, cuda or cpu')

    args = parser.parse_args()

    device = args.device

    if args.mech == "original":
        for window_size in [32]:
            process_and_evaluate(
                args=args,
                window_size=window_size,
                mech="original",
                device=device,
                # num_landmarks & iters are ignored when mech=='original'
                num_landmarks=None,
                iters=None,
                sample_k=args.sample_k
            )

    elif args.mech == "pnp_nystra":
        for window_size in [32]:
            for num_landmarks in [16]:
                for iters in [2]:
                    process_and_evaluate(
                        args=args,
                        window_size=window_size,
                        mech="pnp_nystra",
                        device=device,
                        num_landmarks=num_landmarks,
                        iters=iters,
                        sample_k= args.sample_k
                    )


def define_model(args, mech, num_landmarks, iters, window_size, img_size):
    model = net(mech = mech, num_landmarks= num_landmarks, iters = iters, upscale=args.scale, in_chans=3, img_size=img_size, window_size=window_size,
                img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
    param_key_g = 'params'

    
    pretrained_model = torch.load(args.model_path)
    state_dict = pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model

    model_state_dict = model.state_dict()

    filtered_state_dict = {}

    for key, value in state_dict.items():
        if key in model_state_dict:
            if value.size() == model_state_dict[key].size():
                filtered_state_dict[key] = value

    model_state_dict.update(filtered_state_dict)
    model.load_state_dict(model_state_dict)

    return model


def setup(args, window_size):
    save_dir = f'results/swinir_x{args.scale}'
    folder = args.folder_gt
    border = args.scale
    window_size = window_size

    return folder, save_dir, border, window_size


def get_image_pair(args, path):
    (imgname, imgext) = os.path.splitext(os.path.basename(path))

    img_gt = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    img_lq = cv2.imread(f'{args.folder_lq}/{imgname}x{args.scale}{imgext}', cv2.IMREAD_COLOR).astype(
        np.float32) / 255.

    return imgname, img_lq, img_gt


def test(img_lq, model, args, window_size):
    # print(model.img_size)
    if args.tile is None:
        # test the image as a whole
        output = model(img_lq)
    else:
        # test the image tile by tile
        b, c, h, w = img_lq.size()
        tile = min(args.tile, h, w)
        assert tile % window_size == 0, "tile size should be a multiple of window_size"
        tile_overlap = args.tile_overlap
        sf = args.scale

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h*sf, w*sf).type_as(img_lq)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
        output = E.div_(W)

    return output

if __name__ == '__main__':
    main()

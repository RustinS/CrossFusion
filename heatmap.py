import math
import os

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import openslide
import torch
from PIL import Image

from models.CrossFusion import CrossFusion
from utils.general_utils import get_eval_args, set_random_seed

Image.MAX_IMAGE_PIXELS = 933120000


def to_percentiles(scores):
    from scipy.stats import rankdata
    scores = rankdata(scores, 'average')/len(scores) * 100
    return scores

def screen_coords(scores, coords, top_left, bot_right):
    bot_right = np.array(bot_right)
    top_left = np.array(top_left)
    mask = np.logical_and(np.all(coords >= top_left, axis=1), np.all(coords <= bot_right, axis=1))
    scores = scores[mask]
    coords = coords[mask]
    return scores, coords

class WholeSlideImage(object):
    def __init__(self, path):
        self.name = os.path.splitext(os.path.basename(path))[0]
        self.wsi = openslide.open_slide(path)
        self.level_downsamples = self._assert_level_downsamples()
        self.level_dim = self.wsi.level_dimensions

        self.contours_tissue = None
        self.contours_tumor = None
        self.hdf5_file = None

    def get_open_slide(self):
        return self.wsi

    def vis_wsi(
        self,
        vis_level=0,
        max_size=None,
        top_left=None,
        bot_right=None,
        custom_downsample=1,
    ):
        downsample = self.level_downsamples[vis_level]
        scale = [1 / downsample[0], 1 / downsample[1]]

        if top_left is not None and bot_right is not None:
            top_left = tuple(top_left)
            bot_right = tuple(bot_right)
            w, h = tuple((np.array(bot_right) * scale).astype(int) - (np.array(top_left) * scale).astype(int))
            region_size = (w, h)
        else:
            top_left = (0, 0)
            region_size = self.level_dim[vis_level]

        img = np.array(self.wsi.read_region(top_left, vis_level, region_size).convert("RGB"))

        img = Image.fromarray(img)

        w, h = img.size
        if custom_downsample > 1:
            img = img.resize((int(w / custom_downsample), int(h / custom_downsample)))

        if max_size is not None and (w > max_size or h > max_size):
            resizeFactor = max_size / w if w > h else max_size / h
            img = img.resize((int(w * resizeFactor), int(h * resizeFactor)))

        return img

    def _assert_level_downsamples(self):
        level_downsamples = []
        dim_0 = self.wsi.level_dimensions[0]

        for downsample, dim in zip(self.wsi.level_downsamples, self.wsi.level_dimensions):
            estimated_downsample = (dim_0[0] / float(dim[0]), dim_0[1] / float(dim[1]))
            level_downsamples.append(estimated_downsample) if estimated_downsample != (
                downsample,
                downsample,
            ) else level_downsamples.append((downsample, downsample))

        return level_downsamples

    def vis_heatmap(
        self,
        scores,
        coords,
        vis_level=-1,
        top_left=None,
        bot_right=None,
        patch_size=(256, 256),
        blank_canvas=False,
        alpha=0.6,
        blur=True,
        overlap=0.0,
        convert_to_percentiles=True,
        max_size=None,
        custom_downsample=1,
        cmap="inferno",
        thresh=0.7,
        binarize=False,
        score_thresh=0.0
    ):
        if vis_level < 0:
            vis_level = self.wsi.get_best_level_for_downsample(32)

        downsample = self.level_downsamples[vis_level]
        scale = [1 / downsample[0], 1 / downsample[1]]

        if len(scores.shape) == 2:
            scores = scores.flatten()

        threshold = (1.0 / len(scores) if thresh < 0 else thresh) if binarize else 0.0

        if top_left is not None and bot_right is not None:
            scores, coords = screen_coords(scores, coords, top_left, bot_right)
            coords = coords - top_left
            top_left = tuple(top_left)
            bot_right = tuple(bot_right)
            w, h = tuple((np.array(bot_right) * scale).astype(int) - (np.array(top_left) * scale).astype(int))
            region_size = (w, h)

        else:
            region_size = self.level_dim[vis_level]
            top_left = (0, 0)
            bot_right = self.level_dim[0]
            w, h = region_size

        patch_size = np.ceil(np.array(patch_size) * np.array(scale)).astype(int)
        coords = np.ceil(coords * np.array(scale)).astype(int)

        print("\ncreating heatmap for: ")
        print("top_left: ", top_left, "bot_right: ", bot_right)
        print("w: {}, h: {}".format(w, h))
        print("scaled patch size: ", patch_size)

        if convert_to_percentiles:
            scores = to_percentiles(scores)

        scores /= 100

        overlay = np.full(np.flip(region_size), 0).astype(float)
        counter = np.full(np.flip(region_size), 0).astype(np.uint16)
        count = 0
        for idx in range(len(coords)):
            score = scores[idx]
            if binarize:
                if score >= threshold:
                    count+=1
            elif score < score_thresh:
                score=0.0
            coord = coords[idx]
            overlay[coord[1] : coord[1] + patch_size[1], coord[0] : coord[0] + patch_size[0]] += score
            counter[coord[1] : coord[1] + patch_size[1], coord[0] : coord[0] + patch_size[0]] += 1

        if binarize:
            print('\nbinarized tiles based on cutoff of {}'.format(threshold))
            print('identified {}/{} patches as positive'.format(count, len(coords)))

        zero_mask = counter == 0

        if binarize:
            overlay[~zero_mask] = np.around(overlay[~zero_mask] / counter[~zero_mask])
        else:
            overlay[~zero_mask] = overlay[~zero_mask] / counter[~zero_mask]

        del counter
        if blur:
            overlay = cv2.GaussianBlur(overlay, tuple((patch_size * (1 - overlap)).astype(int) * 2 + 1), 0)

        if not blank_canvas:
            img = np.array(self.wsi.read_region(top_left, vis_level, region_size).convert("RGB"))
        else:
            img = np.array(Image.new(size=region_size, mode="RGB", color=(255, 255, 255)))

        print("\ncomputing heatmap image")
        print("total of {} patches".format(len(coords)))
        twenty_percent_chunk = max(1, int(len(coords) * 0.2))

        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)

        for idx in range(len(coords)):
            if (idx + 1) % twenty_percent_chunk == 0:
                print("progress: {}/{}".format(idx, len(coords)))

            score = scores[idx]
            coord = coords[idx]
            if score >= threshold:
                raw_block = overlay[coord[1] : coord[1] + patch_size[1], coord[0] : coord[0] + patch_size[0]]
                if np.all(raw_block < 0.5):
                    img_block = img[coord[1] : coord[1] + patch_size[1], coord[0] : coord[0] + patch_size[0]].copy()
                else:
                    img_block = (cmap(raw_block) * 255)[:, :, :3].astype(np.uint8)


                img[coord[1] : coord[1] + patch_size[1], coord[0] : coord[0] + patch_size[0]] = img_block.copy()

        print("Done")
        del overlay

        if blur:
            img = cv2.GaussianBlur(img, tuple((patch_size * (1 - overlap)).astype(int) * 2 + 1), 0)

        if alpha < 1.0:
            img = self.block_blending(img, vis_level, top_left, bot_right, alpha=alpha, blank_canvas=blank_canvas, block_size=1024)

        img = Image.fromarray(img)
        w, h = img.size

        if custom_downsample > 1:
            img = img.resize((int(w / custom_downsample), int(h / custom_downsample)))

        if max_size is not None and (w > max_size or h > max_size):
            resizeFactor = max_size / w if w > h else max_size / h
            img = img.resize((int(w * resizeFactor), int(h * resizeFactor)))

        return img

    def block_blending(self, img, vis_level, top_left, bot_right, alpha=0.5, blank_canvas=False, block_size=1024):
        print("\ncomputing blend")
        downsample = self.level_downsamples[vis_level]
        w = img.shape[1]
        h = img.shape[0]
        block_size_x = min(block_size, w)
        block_size_y = min(block_size, h)
        print("using block size: {} x {}".format(block_size_x, block_size_y))

        shift = top_left
        for x_start in range(top_left[0], bot_right[0], block_size_x * int(downsample[0])):
            for y_start in range(top_left[1], bot_right[1], block_size_y * int(downsample[1])):

                x_start_img = int((x_start - shift[0]) / int(downsample[0]))
                y_start_img = int((y_start - shift[1]) / int(downsample[1]))

                y_end_img = min(h, y_start_img + block_size_y)
                x_end_img = min(w, x_start_img + block_size_x)

                if y_end_img == y_start_img or x_end_img == x_start_img:
                    continue

                blend_block = img[y_start_img:y_end_img, x_start_img:x_end_img]
                blend_block_size = (x_end_img - x_start_img, y_end_img - y_start_img)

                if not blank_canvas:
                    pt = (x_start, y_start)
                    canvas = np.array(self.wsi.read_region(pt, vis_level, blend_block_size).convert("RGB"))
                else:
                    canvas = np.array(Image.new(size=blend_block_size, mode="RGB", color=(255, 255, 255)))

                img[y_start_img:y_end_img, x_start_img:x_end_img] = cv2.addWeighted(blend_block, alpha, canvas, 1 - alpha, 0)
        return img


def build_model(opts, save_dir):
    model = CrossFusion(
        embed_dim=opts.embed_dim,
        num_heads=opts.num_heads,
        num_layers=opts.num_attn_layers,
        backbone_dim=opts.backbone_dim,
        n_classes=opts.n_classes,
    )

    state_dict = torch.load(os.path.join(save_dir, "best_model.pt"), weights_only=True)

    for key in list(state_dict.keys()):
        if "coattn" in key:
            state_dict[key.replace("coattn", "cross_attn")] = state_dict.pop(key)
    for key in list(state_dict.keys()):
        if "plain" in key:
            state_dict[key.replace("plain", "source")] = state_dict.pop(key)
    for key in list(state_dict.keys()):
        if "pos_embedding" in key:
            state_dict[key.replace("pos_embedding", "ppeg")] = state_dict.pop(key)
    for key in list(state_dict.keys()):
        if "_norm" in key:
            state_dict[key.replace("_norm", "_ln")] = state_dict.pop(key)

    model.load_state_dict(state_dict)
    model = model.to("cuda")
    model = torch.nn.DataParallel(model)

    return model

def compute_patch_importance(attn_weights):
    avg_attn = attn_weights.mean(dim=1)

    patch_importance = avg_attn.sum(dim=1)

    return patch_importance.cpu().numpy()

def compute_patch_importance_cross(attn_weights):
    avg_attn = attn_weights.mean(dim=1)

    importance_N = avg_attn.sum(dim=2)
    importance_N_prime = avg_attn.sum(dim=1)

    return importance_N.cpu().numpy(), importance_N_prime.cpu().numpy()

def create_heatmaps(opts):
    image_file_name = "TCGA-A1-A0SP-01Z-00-DX1.20D689C6-EFA5-4694-BE76-24475A89ACC0"
    uid = "aa8c4f21-2fce-4ccb-b270-4a1ad633f36e"
    fold_id = 3
    dataset = "BRCA"

    split_save_dir = os.path.join(opts.save_dir, f"fold_{fold_id}")

    opts.n_classes = 4
    model = build_model(opts, split_save_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    pt_base_path = f"/projects/patho5nobackup/TCGA/Survival_Data/{dataset}/features_clam/resnet50"
    h5_base_path = f"/projects/patho5nobackup/TCGA/Survival_Data/{dataset}/patches_clam"

    x5_pt_path = os.path.join(pt_base_path, f"x5/pt_files/{image_file_name}.pt")
    x10_pt_path = os.path.join(pt_base_path, f"x10/pt_files/{image_file_name}.pt")
    x20_pt_path = os.path.join(pt_base_path, f"x20/pt_files/{image_file_name}.pt")

    x5_patches = torch.load(x5_pt_path, map_location="cpu", weights_only=False).to(device).unsqueeze(0)
    x10_patches = torch.load(x10_pt_path, map_location="cpu", weights_only=False).to(device).unsqueeze(0)
    x20_patches = torch.load(x20_pt_path, map_location="cpu", weights_only=False).to(device).unsqueeze(0)

    with torch.no_grad():
        _, _, _, _, attention_scores = model(x5_patches, x10_patches, x20_patches, attn_weights=True)

    x20_h5_path = os.path.join(h5_base_path, f"x20/patches/{image_file_name}.h5")
    x10_h5_path = os.path.join(h5_base_path, f"x10/patches/{image_file_name}.h5")
    x5_h5_path = os.path.join(h5_base_path, f"x5/patches/{image_file_name}.h5")

    WSI_object = WholeSlideImage(f"/projects/patho5/Survival/TCGA/BRCA/{uid}/{image_file_name}.svs")

    with h5py.File(x20_h5_path, "r") as h5_file:
        x20_coords = h5_file["coords"][:]

    with h5py.File(x10_h5_path, "r") as h5_file:
        x10_coords = h5_file["coords"][:]

    with h5py.File(x5_h5_path, "r") as h5_file:
        x5_coords = h5_file["coords"][:]

    layer_name = "fine_cross_attn_w"
    # place = "post"
    # final_heatmap = compute_patch_importance(attention_scores[layer_name][place], multihead=False)

    final_heatmap, final_heatmap_aux = compute_patch_importance_cross(attention_scores[layer_name])

    heatmap = WSI_object.vis_heatmap(scores=final_heatmap, coords=x10_coords, patch_size=(1024, 1024))
    heatmap.save(f"heatmap_{layer_name}.png")

    heatmap = WSI_object.vis_heatmap(scores=final_heatmap_aux, coords=x20_coords, patch_size=(512, 512))
    heatmap.save(f"heatmap_{layer_name}_x10.png")

if __name__ == "__main__":
    args = get_eval_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_random_seed(args.random_seed, device)
    create_heatmaps(args)

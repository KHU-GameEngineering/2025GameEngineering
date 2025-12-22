import os
import argparse
import numpy as np
import torch
import rembg
from PIL import Image
from torchvision.transforms import v2
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

from src.utils.preprocess import preprocess_image
from src.utils.train_util import instantiate_from_config
from src.utils.camera_util import (
    FOV_to_intrinsics,
    get_zero123plus_input_cameras,
    get_circular_camera_poses,
)
from src.utils.mesh_util import save_obj, save_obj_with_mtl
from src.utils.infer_util import  save_video

def preprocess_for_zero123(
    path_in: str,
    target_size: int = 512,
    use_rembg: bool = True,
    bg_color=(255, 255, 255),
    margin_ratio: float = 0.08,
) -> Image.Image:

    img = Image.open(path_in).convert("RGBA")

    # 1. 배경 제거
    if use_rembg:
        img = rembg.remove(img)

    arr = np.array(img)
    h, w, c = arr.shape
    alpha = arr[..., 3]

    ys, xs = np.where(alpha > 0)

    if len(xs) == 0:
        side = min(w, h)
        x0 = (w - side) // 2
        y0 = (h - side) // 2
        arr = arr[y0:y0+side, x0:x0+side]
    else:
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        bw = x_max - x_min + 1
        bh = y_max - y_min + 1
        mx = int(bw * margin_ratio)
        my = int(bh * margin_ratio)

        x0 = max(0, x_min + mx)
        x1 = min(w, x_max - mx)
        y0 = max(0, y_min + my)
        y1 = min(h, y_max - my)

        arr = arr[y0:y1, x0:x1]
    cropped = Image.fromarray(arr, mode="RGBA")
    
    bg = Image.new("RGBA", cropped.size, (*bg_color, 255))
    rgb = Image.alpha_composite(bg, cropped).convert("RGB")
    cw, ch = rgb.size
    side = max(cw, ch)
    canvas = Image.new("RGB", (side, side), bg_color)
    offset = ((side - cw) // 2, (side - ch) // 2)
    canvas.paste(rgb, offset)
    canvas = canvas.resize((target_size, target_size), Image.BICUBIC)
    return canvas

def get_render_cameras(
    batch_size=1,
    M=120,
    radius=4.0,
    elevation=20.0,
    is_flexicubes=False,
):
    """Generate rendering cameras on a circular trajectory."""
    c2ws = get_circular_camera_poses(M=M, radius=radius, elevation=elevation)

    if is_flexicubes:
        cameras = torch.linalg.inv(c2ws)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    else:
        extrinsics = c2ws.flatten(-2)
        intrinsics = (
            FOV_to_intrinsics(30.0)
            .unsqueeze(0)
            .repeat(M, 1, 1)
            .float()
            .flatten(-2)
        )
        cameras = torch.cat([extrinsics, intrinsics], dim=-1)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1)
    return cameras


def crop_six_views(grid: Image.Image):
    """Split 3x2 grid from Zero123Plus into 6 views (PIL images)."""
    w, h = grid.size
    tile_w = w // 2
    tile_h = h // 3

    coords = [
        (0, 0, tile_w, tile_h),
        (tile_w, 0, w, tile_h),
        (0, tile_h, tile_w, 2 * tile_h),
        (tile_w, tile_h, w, 2 * tile_h),
        (0, 2 * tile_h, tile_w, h),
        (tile_w, 2 * tile_h, w, h),
    ]
    return [grid.crop(c) for c in coords]


def render_frames(
    model,
    planes,
    render_cameras,
    render_size=512,
    chunk_size=1,
    is_flexicubes=False,
):
    """Render frames from triplanes along a camera trajectory."""
    frames = []
    num = render_cameras.shape[1]

    for i in tqdm(range(0, num, chunk_size)):
        cams = render_cameras[:, i : i + chunk_size]
        if is_flexicubes:
            out = model.forward_geometry(
                planes,
                cams,
                render_size=render_size,
            )["img"]
        else:
            out = model.forward_synthesizer(
                planes,
                cams,
                render_size=render_size,
            )["images_rgb"]
        frames.append(out)

    frames = torch.cat(frames, dim=1)[0]
    return frames

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to config file.")
    parser.add_argument("input_path", type=str, help="Path to input image or directory.")
    parser.add_argument("--output_path", type=str, default="outputs/", help="Output directory.")
    parser.add_argument("--diffusion_steps", type=int, default=75, help="Denoising sampling steps for Zero123Plus.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale of generated object.")
    parser.add_argument("--distance", type=float, default=4.5, help="Render distance for video.")
    parser.add_argument("--view", type=int, default=6, choices=[4, 6], help="Number of input views.")
    parser.add_argument("--no_rembg", action="store_true", help="Do NOT remove background of the input image.")
    parser.add_argument("--export_texmap", action="store_true", help="Export mesh with texture map (OBJ+MTL).")
    parser.add_argument("--save_video", action="store_true", help="Save a circular-view video.")
    return parser

def run(args):
   
    print("[run_original] export_texmap =", args.export_texmap)

    seed_everything(args.seed)

    config = OmegaConf.load(args.config)
    config_name = os.path.basename(args.config).replace(".yaml", "")
    model_config = config.model_config
    infer_config = config.infer_config

    IS_FLEXICUBES = config_name.startswith("instant-mesh")
    device = torch.device("cuda")

    print("Loading diffusion model (Zero123Plus) ...")
    pipeline = DiffusionPipeline.from_pretrained(
    "sudo-ai/zero123plus-v1.2",
    custom_pipeline="sudo-ai/zero123plus-pipeline",
    torch_dtype=torch.float16,
)


    model_path = r"C:\Users\chaeyeonhan\GE\InstantMesh\ckpts\loss=0.0144.ckpt"
    print(f"Loading checkpoint from: {model_path}")
    state_dict = torch.load(model_path)
    pipeline.unet.load_state_dict(state_dict, strict=False)

    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipeline.scheduler.config, timestep_spacing="trailing"
    )
    pipeline = pipeline.to(device)

    print("Loading custom white-background unet ...")
    if os.path.exists(infer_config.unet_path):
        unet_ckpt_path = infer_config.unet_path
    else:
        unet_ckpt_path = hf_hub_download(
            repo_id="TencentARC/InstantMesh",
            filename="diffusion_pytorch_model.bin",
            repo_type="model",
        )
    state_dict = torch.load(unet_ckpt_path, map_location="cpu")
    pipeline.unet.load_state_dict(state_dict, strict=True)
    pipeline = pipeline.to(device)

    model = instantiate_from_config(model_config)
    if os.path.exists(infer_config.model_path):
        model_ckpt_path = infer_config.model_path
    else:
        model_ckpt_path = hf_hub_download(
            repo_id="TencentARC/InstantMesh",
            filename=f"{config_name.replace('-', '_')}.ckpt",
            repo_type="model",
        )
    state_dict = torch.load(model_ckpt_path, map_location="cpu")["state_dict"]
    state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith("lrm_generator.")}
    model.load_state_dict(state_dict, strict=True)

    model = model.to(device)
    if IS_FLEXICUBES:
        model.init_flexicubes_geometry(device, fovy=30.0)
    model.eval()

    image_path = os.path.join(args.output_path, config_name, "images")
    mesh_path  = os.path.join(args.output_path, config_name, "meshes")
    video_path = os.path.join(args.output_path, config_name, "videos")
    os.makedirs(image_path, exist_ok=True)
    os.makedirs(mesh_path, exist_ok=True)
    os.makedirs(video_path, exist_ok=True)

    if os.path.isdir(args.input_path):
        input_files = [
            os.path.join(args.input_path, f)
            for f in os.listdir(args.input_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        ]
    else:
        input_files = [args.input_path]

    print(f"Total number of input images: {len(input_files)}")

    outputs = []
    grid_paths = []
    view_paths = []   # 각 이미지별 view0~5 경로들
    input_pre_paths = []

    # ---------- Zero123Plus 멀티뷰 생성 ----------
    for idx, image_file in enumerate(input_files):
        name = os.path.basename(image_file).rsplit(".", 1)[0]
        print(f"[{idx+1}/{len(input_files)}] Imagining {name} ...")

        img = preprocess_for_zero123(
            image_file, target_size=512, use_rembg=not args.no_rembg
        )

        debug_in_path = os.path.join(image_path, f"{name}_input_preprocessed.png")
        img.save(debug_in_path)
        input_pre_paths.append(debug_in_path)

        result = pipeline(img, num_inference_steps=args.diffusion_steps).images[0]

        grid_path = os.path.join(image_path, f"{name}_grid.png")
        result.save(grid_path)
        grid_paths.append(grid_path)
        print(f"Grid image saved to {grid_path}")

        views = crop_six_views(result)

        per_img_view_paths = []
        view_tensors = []
        for i, v in enumerate(views):
            v = v.convert("RGB")

            debug_path = os.path.join(image_path, f"{name}_view{i}.png")
            v.save(debug_path)
            per_img_view_paths.append(debug_path)

            v_np = np.asarray(v, dtype=np.float32) / 255.0
            v_tensor = torch.from_numpy(v_np).permute(2, 0, 1)
            view_tensors.append(v_tensor)

        view_paths.append(per_img_view_paths)
        images = torch.stack(view_tensors, dim=0)
        outputs.append({"name": name, "images": images})

    # pipeline 메모리 정리
    del pipeline
    torch.cuda.empty_cache()

    # ---------- InstantMesh 메쉬 생성 ----------
    base_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0 * args.scale).to(device)
    chunk_size = 20 if IS_FLEXICUBES else 1

    mesh_paths = []
    video_paths = []

    for idx, sample in enumerate(outputs):
        name = sample["name"]
        print(f"[{idx+1}/{len(outputs)}] Creating {name} ...")

        imgs = sample["images"].unsqueeze(0).to(device)
        cams = base_cameras.clone()

        if args.view == 4:
            indices = torch.tensor([0, 2, 4, 5], device=device).long()
            imgs = imgs[:, indices]
            cams = cams[:, indices]

        with torch.no_grad():
            planes = model.forward_planes(imgs, cams)

            mesh_out_path = os.path.join(mesh_path, f"{name}.obj")
            mesh_out = model.extract_mesh(planes, use_texture_map=args.export_texmap, **infer_config)

            if args.export_texmap:
                vertices, faces, uvs, mesh_tex_idx, tex_map = mesh_out
                save_obj_with_mtl(
                    vertices.cpu().numpy(),
                    uvs.cpu().numpy(),
                    faces.cpu().numpy(),
                    mesh_tex_idx.cpu().numpy(),
                    tex_map.permute(1, 2, 0).cpu().numpy(),
                    mesh_out_path,
                )
            else:
                vertices, faces, vertex_colors = mesh_out
                save_obj(vertices, faces, vertex_colors, mesh_out_path)

            print(f"Mesh saved to {mesh_out_path}")
            mesh_paths.append(mesh_out_path)

            if args.save_video:
                video_out_path = os.path.join(video_path, f"{name}.mp4")
                render_size = infer_config.render_resolution
                render_cameras = get_render_cameras(
                    batch_size=1, M=120, radius=args.distance, elevation=20.0,
                    is_flexicubes=IS_FLEXICUBES
                ).to(device)

                frames = render_frames(
                    model, planes, render_cameras=render_cameras, render_size=render_size,
                    chunk_size=chunk_size, is_flexicubes=IS_FLEXICUBES
                )
                save_video(frames, video_out_path, fps=30)
                print(f"Video saved to {video_out_path}")
                video_paths.append(video_out_path)

    return {
        "config_name": config_name,
        "input_files": input_files,
        "input_preprocessed": input_pre_paths,
        "grid_paths": grid_paths,
        "view_paths": view_paths,
        "mesh_paths": mesh_paths,
        "video_paths": video_paths,
        "output_root": args.output_path,
    }


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run(args)
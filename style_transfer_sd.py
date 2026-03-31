import argparse
import os
import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler


def get_canny_edges(image_path, size, low=50, high=150):
    img   = cv2.imread(image_path)
    img   = cv2.resize(img, size)
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low, high)
    return Image.fromarray(np.stack([edges] * 3, axis=-1))


def get_output_size(image_path, max_dim=512):
    img = Image.open(image_path)
    orig_w, orig_h = img.size
    scale = max_dim / max(orig_w, orig_h)
    new_w = (int(orig_w * scale) // 64) * 64
    new_h = (int(orig_h * scale) // 64) * 64
    return max(new_w, 64), max(new_h, 64)


def decode_latent(pipe, latent):
    """
    Convert a latent tensor to a PIL image.
    Used by the timelapse callback to visualise each denoising step.
    The VAE decoder maps from latent space (4 channels) to pixel space (3 channels).
    """
    with torch.no_grad():
        scaled = latent / pipe.vae.config.scaling_factor
        decoded = pipe.vae.decode(scaled).sample
        decoded = (decoded / 2 + 0.5).clamp(0, 1)
        decoded = decoded.cpu().permute(0, 2, 3, 1).numpy()
        return Image.fromarray((decoded[0] * 255).astype(np.uint8))


def make_timelapse_gif(frames_dir, output_path, fps=8):
    """Combine saved step frames into an animated GIF."""
    frames = sorted([
        os.path.join(frames_dir, f)
        for f in os.listdir(frames_dir)
        if f.endswith(".png")
    ])
    if not frames:
        print("No frames found for timelapse.")
        return

    images = [Image.open(f).convert("RGB") for f in frames]
    duration = int(1000 / fps)

    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
    )
    print(f"Timelapse GIF saved → {output_path}  ({len(images)} frames)")


def load_pipeline(device="cuda"):
    dtype = torch.float16 if device == "cuda" else torch.float32

    print("Loading ControlNet (canny)...")
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_canny",
        torch_dtype=dtype,
    )

    print("Loading Stable Diffusion img2img pipeline...")
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "Lykon/dreamshaper-8",
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
    ).to(device)

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()

    print("Loading IP-Adapter...")
    pipe.load_ip_adapter(
        "ip_adapter_files",
        subfolder="models",
        weight_name="ip-adapter_sd15.bin",
        image_encoder_folder="models/image_encoder",
    )

    return pipe


def run(
    content_path,
    style_path,
    output_path       = "result.png",
    resolution        = 512,
    controlnet_weight = 1.0,
    ip_adapter_weight = 0.8,
    denoise           = 0.75,
    num_steps         = 30,
    seed              = 42,
    canny_low         = 50,
    canny_high        = 150,
    save_edges        = True,
    timelapse         = False,
    timelapse_fps     = 8,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    out_w, out_h = get_output_size(content_path, resolution)
    print(f"Output size: {out_w}x{out_h}")

    print(f"Extracting edges from {content_path}...")
    canny_image = get_canny_edges(content_path, (out_w, out_h), canny_low, canny_high)
    if save_edges:
        edges_path = output_path.replace(".png", "_edges.png")
        canny_image.save(edges_path)
        print(f"  Edges saved → {edges_path}")

    content_image = Image.open(content_path).convert("RGB").resize((out_w, out_h))
    style_image   = Image.open(style_path).convert("RGB").resize((out_w, out_h))

    pipe = load_pipeline(device)
    pipe.set_ip_adapter_scale(ip_adapter_weight)

    print(f"Generating at {out_w}x{out_h}, {num_steps} steps...")
    print(f"  ControlNet weight : {controlnet_weight}")
    print(f"  IP-Adapter weight : {ip_adapter_weight}")
    print(f"  Denoise strength  : {denoise}")

    generator = torch.Generator(device=device).manual_seed(seed)

    frames_dir = None
    callback   = None

    if timelapse:
        frames_dir = output_path.replace(".png", "_frames")
        os.makedirs(frames_dir, exist_ok=True)
        print(f"  Timelapse frames → {frames_dir}/")

        def callback(pipe, step, timestep, kwargs):
            """
            Called by diffusers after every denoising step.
            Decodes the current latent to a pixel image and saves it.
            This shows the image evolving from noise → final result.
            """
            latents = kwargs.get("latents")
            if latents is not None:
                frame = decode_latent(pipe, latents)
                frame_path = os.path.join(frames_dir, f"step_{step:03d}.png")
                frame.save(frame_path)
            return kwargs

    result = pipe(
        prompt="",
        negative_prompt="hat, cap, headwear, blurry, deformed, ugly, bad anatomy, watermark, low quality",
        image=content_image,
        control_image=canny_image,
        ip_adapter_image=style_image,
        num_inference_steps=num_steps,
        strength=denoise,
        controlnet_conditioning_scale=controlnet_weight,
        generator=generator,
        callback_on_step_end=callback,
    ).images[0]

    result.save(output_path)
    print(f" Done! Saved → {output_path}")


    comparison_path = output_path.replace(".png", "_comparison.png")
    comparison = Image.new("RGB", (out_w * 3, out_h))
    comparison.paste(content_image, (0, 0))
    comparison.paste(style_image,   (out_w, 0))
    comparison.paste(result,        (out_w * 2, 0))
    comparison.save(comparison_path)
    print(f"Comparison → {comparison_path}  (content | style | result)")


    if timelapse and frames_dir:
        gif_path = output_path.replace(".png", "_timelapse.gif")
        make_timelapse_gif(frames_dir, gif_path, timelapse_fps)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Style transfer: apply style of one image to another"
    )
    parser.add_argument("--content",            required=True)
    parser.add_argument("--style",              required=True)
    parser.add_argument("--output",             default="result.png")
    parser.add_argument("--resolution",         type=int,   default=512)
    parser.add_argument("--controlnet-weight",  type=float, default=1.0)
    parser.add_argument("--ip-adapter-weight",  type=float, default=0.8)
    parser.add_argument("--denoise",            type=float, default=0.75)
    parser.add_argument("--steps",              type=int,   default=30)
    parser.add_argument("--seed",               type=int,   default=42)
    parser.add_argument("--canny-low",          type=int,   default=50)
    parser.add_argument("--canny-high",         type=int,   default=150)
    parser.add_argument("--no-save-edges",      action="store_true")
    parser.add_argument("--timelapse",          action="store_true",
                        help="Save a frame at each step and combine into a GIF")
    parser.add_argument("--timelapse-fps",      type=int,   default=8,
                        help="GIF playback speed in frames per second (default 8)")
    args = parser.parse_args()

    run(
        content_path      = args.content,
        style_path        = args.style,
        output_path       = args.output,
        resolution        = args.resolution,
        controlnet_weight = args.controlnet_weight,
        ip_adapter_weight = args.ip_adapter_weight,
        denoise           = args.denoise,
        num_steps         = args.steps,
        seed              = args.seed,
        canny_low         = args.canny_low,
        canny_high        = args.canny_high,
        save_edges        = not args.no_save_edges,
        timelapse         = args.timelapse,
        timelapse_fps     = args.timelapse_fps,
    )
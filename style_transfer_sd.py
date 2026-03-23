import argparse
import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image

#Generates edges for structure

def get_canny_edges(image_path: str, size: tuple,
                    low: int = 50, high: int = 150) -> Image.Image:

    img   = cv2.imread(image_path)
    img   = cv2.resize(img, size)  
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low, high)
    return Image.fromarray(np.stack([edges] * 3, axis=-1))


def get_output_size(image_path: str, max_dim: int = 512) -> tuple:

    img    = Image.open(image_path)
    orig_w, orig_h = img.size
    scale  = max_dim / max(orig_w, orig_h)
    new_w  = (int(orig_w * scale) // 64) * 64
    new_h  = (int(orig_h * scale) // 64) * 64
    new_w  = max(new_w, 64)
    new_h  = max(new_h, 64)
    return new_w, new_h


def load_pipeline(device: str = "cuda"):
   
    dtype = torch.float16 if device == "cuda" else torch.float32

    print("Loading ControlNet (canny)...")
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_canny",
        torch_dtype=dtype,
    )

    print("Loading Stable Diffusion img2img pipeline...")
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
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
    content_path:      str,
    style_path:        str,
    output_path:       str   = "result.png",
    resolution:        int   = 512,
    controlnet_weight: float = 1.0,
    ip_adapter_weight: float = 0.8,
    denoise:           float = 0.75,
    num_steps:         int   = 30,
    seed:              int   = 42,
    canny_low:         int   = 50,
    canny_high:        int   = 150,
    save_edges:        bool  = True,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    out_w, out_h = get_output_size(content_path, resolution)
    print(f"\nContent image size → output size: {out_w}x{out_h}")

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

    print(f"\nGenerating at {out_w}x{out_h}, {num_steps} steps...")
    print(f"  ControlNet weight : {controlnet_weight}  (structure)")
    print(f"  IP-Adapter weight : {ip_adapter_weight}  (style)")
    print(f"  Denoise strength  : {denoise}  (higher = more style)")

    generator = torch.Generator(device=device).manual_seed(seed)

    result = pipe(
        prompt="",
        negative_prompt="",
        image=content_image,
        control_image=canny_image,
        ip_adapter_image=style_image,
        num_inference_steps=num_steps,
        strength=denoise,
        controlnet_conditioning_scale=controlnet_weight,
        generator=generator,
    ).images[0]

    result.save(output_path)
    print(f"\nDone! Saved → {output_path}")

    comparison_path = output_path.replace(".png", "_comparison.png")
    comparison = Image.new("RGB", (out_w * 3, out_h))
    comparison.paste(content_image, (0, 0))
    comparison.paste(style_image,   (out_w, 0))
    comparison.paste(result,        (out_w * 2, 0))
    comparison.save(comparison_path)
    print(f"Comparison → {comparison_path}  (content | style | result)")

    return result



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Style transfer: apply style of one image to another"
    )
    parser.add_argument("--content",            required=True,
                        help="Path to content image (your photo)")
    parser.add_argument("--style",              required=True,
                        help="Path to style image (painting/artwork)")
    parser.add_argument("--output",             default="result.png",
                        help="Output path (default: result.png)")
    parser.add_argument("--resolution",         type=int,   default=512,
                        help="Max dimension — aspect ratio is preserved")
    parser.add_argument("--controlnet-weight",  type=float, default=1.0,
                        help="Structure preservation (0.5-1.5)")
    parser.add_argument("--ip-adapter-weight",  type=float, default=0.8,
                        help="Style strength from image (0.3-1.0)")
    parser.add_argument("--denoise",            type=float, default=0.75,
                        help="0=no change, 1=full generation")
    parser.add_argument("--steps",              type=int,   default=30,
                        help="Diffusion steps (more = better, slower)")
    parser.add_argument("--seed",               type=int,   default=42,
                        help="Random seed — change for variations")
    parser.add_argument("--canny-low",          type=int,   default=50,
                        help="Lower = more edges captured")
    parser.add_argument("--canny-high",         type=int,   default=150)
    parser.add_argument("--no-save-edges",      action="store_true",
                        help="Skip saving edge debug image")
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
    )
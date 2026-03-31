# Style Transfer

Apply the painting style of one image to another.

---

## Deafault Docker command

```powershell
docker run --rm --gpus all `
  -v ${PWD}/input:/app/input `
  -v ${PWD}/output:/app/output `
  -v style_transfer_models:/app/models `
  style-transfer:latest `
  --content /app/input/content.jpg `
  --style /app/input/style.jpg `
  --output /app/output/result.png `
  --steps 30
```

### Command explained

| Part                                   | Description                                                                                                                                                                                              |
| -------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `docker run`                           | Creates and starts a container from an image                                                                                                                                                             |
| `--rm`                                 | Automatically deletes the container when it finishes. Without this, stopped containers pile up on your disk. In the future we could make it into an API endpoint so we dont need to recreate containers. |
| `--gpus all`                           | Gives the container access to NVIDIA GPUs.                                                                                                                                                               |
| `-v ${PWD}/input:/app/input`           | Mounts your local `input` folder into the container at `/app/input`. This is how the container sees your images — without this `/app/input` is empty. The name of the local folder could be changed.     |
| `-v ${PWD}/output:/app/output`         | Same for output — results saved to `/app/output` inside the container appear in your local `output` folder. The name of the local folder could be changed.                                               |
| `-v style_transfer_models:/app/models` | Named Docker volume for model cache. Models download here on first run and persist permanently.                                                                                                          |
| `style-transfer:latest`                | Which image to run. `style-transfer` is the name, `latest` is the tag.                                                                                                                                   |
| `--content /app/input/content.jpg`     | Path to your photo inside the container.                                                                                                                                                                 |
| `--style /app/input/style.jpg`         | Path to your painting inside the container.                                                                                                                                                              |
| `--output /app/output/result.png`      | Where to save the result inside the container — appears in your local `output` folder.                                                                                                                   |

---

## Arguments

| Argument              | Type  | Default      | Min   | Max          | Description                                                                                                                          |
| --------------------- | ----- | ------------ | ----- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------ |
| `--content`           | path  | required     | —     | —            | Path to your photo                                                                                                                   |
| `--style`             | path  | required     | —     | —            | Path to the painting/artwork                                                                                                         |
| `--output`            | path  | `result.png` | —     | —            | Where to save the result                                                                                                             |
| `--resolution`        | int   | `512`        | `64`  | `768`        | Max output dimension. Aspect ratio is preserved. Use `384` for low VRAM, `768` for more detail. Hard limit at 768 for 4GB VRAM       |
| `--controlnet-weight` | float | `1.0`        | `0.0` | `2.0`        | How rigid the face/body structure stays. Higher = more rigid. Above 2.0 face becomes a rigid sketch with no style                    |
| `--ip-adapter-weight` | float | `0.8`        | `0.0` | `1.0`        | How strongly the style image drives the output. Hard limit at 1.0 — above this throws an error                                       |
| `--denoise`           | float | `0.75`       | `0.0` | `1.0`        | How much to repaint. `0.0` = no change, `1.0` = full repaint. Hard limit — cannot go outside 0–1. Actual steps run = denoise × steps |
| `--steps`             | int   | `30`         | `1`   | `100`        | Diffusion steps. More = better quality but slower. No visible improvement above 100                                                  |
| `--seed`              | int   | `42`         | `0`   | `2147483647` | Random seed. Change for different variations. Above max crashes torch                                                                |
| `--canny-low`         | int   | `50`         | `0`   | `254`        | Canny edge lower threshold. Lower = more edges. Must be lower than `--canny-high`                                                    |
| `--canny-high`        | int   | `150`        | `1`   | `255`        | Canny edge upper threshold. Must be higher than `--canny-low`                                                                        |
| `--no-save-edges`     | flag  | off          | —     | —            | Skip saving the edge debug image                                                                                                     |

### Hard limits — will crash if exceeded

| Argument                        | Limit            | Error                 |
| ------------------------------- | ---------------- | --------------------- |
| `--ip-adapter-weight`           | max `1.0`        | diffusers API error   |
| `--denoise`                     | `0.0`–`1.0`      | scheduler error       |
| `--seed`                        | max `2147483647` | torch generator error |
| `--canny-low` >= `--canny-high` | must be lower    | OpenCV error          |

--- | |

### Hard limits — will crash if exceeded

| Argument                        | Limit            | Error                 |
| ------------------------------- | ---------------- | --------------------- |
| `--ip-adapter-weight`           | max `1.0`        | diffusers API error   |
| `--denoise`                     | `0.0`–`1.0`      | scheduler error       |
| `--seed`                        | max `2147483647` | torch generator error |
| `--canny-low` >= `--canny-high` | must be lower    | OpenCV error          |

---

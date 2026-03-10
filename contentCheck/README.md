# contentcheck

Detect abnormal human anatomy in video frames using multiple detection models.

Designed to catch AI-generated video artifacts: extra fingers, impossible joint angles, disproportionate limbs, and other anatomical anomalies.

## Installation

```bash
pip install -r requirements.txt
```

## API Keys

Create an `api_keys.json` file in the `contentCheck/` directory (git-ignored):

```json
{
  "gemini": "YOUR_GEMINI_API_KEY",
  "grok": "YOUR_GROK_API_KEY"
}
```

Keys are resolved in order: CLI flag > `api_keys.json` > environment variable.

## Usage

```bash
# Run with Gemini (default — reads key from api_keys.json)
python -m contentcheck video.mp4

# Run with local models (no API key needed)
python -m contentcheck video.mp4 --models mediapipe yolo

# Combine models
python -m contentcheck video.mp4 --models mediapipe llm-gemini

# Custom threshold and output directory
python -m contentcheck video.mp4 --threshold 0.4 --output results/

# Override API key via CLI
python -m contentcheck video.mp4 --gemini-api-key YOUR_KEY
```

## Available Models

| Model | Key | What it checks | Needs API key? |
|-------|-----|----------------|----------------|
| **MediaPipe** | `mediapipe` | Hand landmarks (finger proportions, joint angles, finger order) + body pose (limb symmetry, joint angles) | No |
| **YOLO11-Pose** | `yolo` | Body keypoints (limb symmetry, joint angles, body proportions) | No |
| **Gemini** | `llm-gemini` | Full visual analysis via Google Gemini vision | Yes (default model) |
| **Grok** | `llm-grok` | Full visual analysis via xAI Grok vision | Yes |

## How It Works

1. **Extract frames** — pulls 1 frame per second from the input video
2. **Run models** — each selected model analyzes every frame independently
3. **Score** — each model returns a score from 0.0 (normal) to 1.0 (clearly abnormal)
4. **Flag & save** — frames exceeding the threshold are saved to the output directory with a detailed report

## Output

- Flagged frames are saved as JPEG files in the output directory (default: `output/`)
- Console output includes a per-frame score table and anomaly details

## Options

```
positional arguments:
  video                 Path to input video file

options:
  --models {mediapipe,yolo,llm-gemini,llm-grok} [...]
                        Which models to run (default: llm-gemini)
  --threshold FLOAT     Score threshold for flagging (default: 0.5)
  --output DIR          Output directory for flagged frames (default: output/)
  --gemini-api-key KEY  Gemini API key
  --grok-api-key KEY    Grok API key
```

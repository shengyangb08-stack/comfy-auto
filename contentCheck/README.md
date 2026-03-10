# contentcheck

Detect abnormal human anatomy in video frames using multiple detection models.

Designed to catch AI-generated video artifacts: extra fingers, impossible joint angles, disproportionate limbs, and other anatomical anomalies.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Run with MediaPipe (default — no API keys needed)
python -m contentcheck video.mp4

# Run with multiple local models
python -m contentcheck video.mp4 --models mediapipe yolo

# Include LLM-based checks (needs API keys)
python -m contentcheck video.mp4 --models mediapipe yolo llm-gemini llm-grok \
    --gemini-api-key YOUR_KEY \
    --grok-api-key YOUR_KEY

# Or set keys as environment variables
export GEMINI_API_KEY=your_key
export GROK_API_KEY=your_key
python -m contentcheck video.mp4 --models llm-gemini llm-grok

# Custom threshold and output directory
python -m contentcheck video.mp4 --threshold 0.4 --output results/
```

## Available Models

| Model | Key | What it checks | Needs API key? |
|-------|-----|----------------|----------------|
| **MediaPipe** | `mediapipe` | Hand landmarks (finger proportions, joint angles, finger order) + body pose (limb symmetry, joint angles) | No |
| **YOLO11-Pose** | `yolo` | Body keypoints (limb symmetry, joint angles, body proportions) | No |
| **Gemini** | `llm-gemini` | Full visual analysis via Google Gemini vision | Yes (`GEMINI_API_KEY`) |
| **Grok** | `llm-grok` | Full visual analysis via xAI Grok vision | Yes (`GROK_API_KEY`) |

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
                        Which models to run (default: mediapipe)
  --threshold FLOAT     Score threshold for flagging (default: 0.5)
  --output DIR          Output directory for flagged frames (default: output/)
  --gemini-api-key KEY  Gemini API key
  --grok-api-key KEY    Grok API key
```

# Real and Apparent Personality Prediction in Human-Human Interaction

## Procedure

1. YOLOX for body detection

  ```bash
  python3 tools/demo_track_personality.py image --path samples --fp16 --fuse --save_result &> outputs/logs/log_samples.txt
  ```

  Note: maintain body relative position by using the largest bounding box to crop the people.

2. Extract R3D features

3. Train model

## Data

### MHHRI

For data information, refer to [data/README.md](./data/README.md)

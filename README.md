# Real and Apparent Personality Prediction in Human-Human Interaction

## Folder Structure

```txt
- data
  :: annotations, data_list, data_summary, hhi_ego_face_fixed, hhi_ego_face_np
  :: hhi_kinect_session_cropped, hhi_kinect_body_np, profiles, README.md
- features
  :: face_features, face_features_fixed, r3d_features, visualize, yolox
- final
- models
- paper
- pretrained
- results
- utils
- visualize
- baseline.py
- train.py
- train_bf.py
- train_bf_ibf.py
- train_scratch.py
- train_scratch_body.py
- train_scratch_face.py
- train_scratch_s.py
- train_scratch_s_body.py
- train_scratch_s_face.py
- vis_feature_map.py
- run.sh
- README.md
```

## Procedure

1. YOLOX for body detection

  ```bash
  python3 tools/demo_track_personality.py image --path samples --fp16 --fuse --save_result &> outputs/logs/log_samples.txt
  ```

  Note: maintain body relative position by using the largest bounding box to crop the people.

2. Train model
  ```bash
  # train.py
  python 
  # train_bf.py
  python 
  # train_bf_ibf.py
  python train_bf_ibf.py --model SBF_nlfc_IBF_nlfc_d_reg --task acq --label reg --trait ALL
  python train_bf_ibf.py --model SBF_nlfc_IBF_nlfc_d_reg --task self --label reg --trait ALL
  python train_bf_ibf.py --model SBF_nlfc_IBF_nlfc_d --task acq --label reg --trait ALL
  python train_bf_ibf.py --model SBF_nlfc_IBF_nlfc_d --task self --label reg --trait ALL
  # train_scratch.py
  python train_scratch.py --model MyModel --task acq --trait ALL
  # train_scratch_s.py
  python train_scratch_s.py --model MyModelS --task acq --trait ALL
  python train_scratch_s.py --model MyModelS --task self --trait ALL
  # train_scratch_s_body.py
  python train_scratch_s_body.py --model MyModelSBody --task acq --trait ALL
  python train_scratch_s_body.py --model MyModelSBody --task self --trait ALL
  # train_scratch_s_face.py
  python train_scratch_s_face.py --model MyModelSFace --task acq --trait ALL
  python train_scratch_s_face.py --model MyModelSFace --task self --trait ALL
  ```

## Data

### MHHRI

For data information, refer to [data/README.md](./data/README.md)

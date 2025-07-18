
# MGVC: MLLM-Guided Video Captioning for the IntentVC Challenge

This repository contains the official implementation of MGVC, the 1st Place solution for the ACM Multimedia Grand Challenge on Intention-Oriented Controllable Video Captioning (IntentVC).
Our code is available at: [MGVC-Intentvc](https://github.com/smuelpeng/MGVC-Intentvc)

If you have any questions or issues, feel free to open an issue on GitHub.

## Data Preparation (Optional)
We preprocess video data in three views: global, temporal-bounding, and center-oriented. The directory structure is as follows:
```
data/intentvc
├── videos           # Original videos and bounding box info
├── videos_center    # Center-cropped videos
└── videos_maxbbox   # Max-bbox cropped videos
```

## Dependencies
Our method is based on the Qwen Docker image. You can launch it with:
```bash
docker run --gpus all --ipc=host --network=host --rm --name qwen2.5 -it qwenllm/qwenvl:2.5-cu121 bash

# Inside the container, navigate to the repo path
pip install -r requirements.txt
export PYTHONPATH=./:$PYTHONPATH
```
For a local environment, all Python dependencies are listed in [requirements.txt.all](requirements.txt.all).

## Model Preparation
Please place the model weights and config files as follows ~~(**Note:** We are currently uploading the required model files. The links and instructions will be updated in this GitHub repository soon.):~~
You can download model weights from DropBox [Model](https://www.dropbox.com/scl/fo/rpqjrrffy5obmpe3w2r92/AHcSFh7IcW8EFeRu-48iZcU?rlkey=u7mr5a2fnljueu4j4ciycua12&st=27nm20ro&dl=0)
```
models/
├── RAG_similarity.pth # RAG sim cache
├── cap_stage1/  # Caption generation model weights
├── cap_stage2/  # Caption generation model weights
├── checkpoint/  # Retrieval/rerank model weights
│   ├── checkpoint_best.pth
│   └── config.yaml
└── reg_model.pth  # Linear regression model
```



## Run Caption Generation
To generate video captions, use the provided script `run_caption.py`. By default, it takes `data/sample_result_public.json` as input and outputs `data/sample_result_public_captioned.json`.
```bash
python run_caption.py
```
- You may modify `SAMPLE_JSON`, `VIDEO_ROOT`, `BBOX_ROOT`, and `MODEL_PATH` in the script as needed.
- The script will automatically call the caption generator and candidate selector modules.

## Run Filtering and Rerank
Filtering and reranking are handled by `core/selector/selector.py`, which can Reproduction our result on leaderboard.


You can run the script as follows:
```bash
python core/selector/selector.py
```

By default, the script will:
- Remove duplicate captions for each video before reranking.
- Use the top-k (default 10, can be changed) captions with the highest TVR scores for further reranking.
- Select the best caption for each video based on a regression model combining TVR, inner BLEU score, and RAG scores.

Key parameters (can be modified in the script):
- `tvr_path`: Path to retrieval model config (e.g., `models/checkpoint/config.yaml`)
- `test_caption_path`: JSON file with candidate captions to rerank
- `train_caption_path`: JSON file with training captions
- `video_root`: Root directory for videos
- `topk`: Number of candidates to consider


## Recommended Hardware
- GPU: 80GB VRAM (e.g., A100/A800)
- Storage: ~100GB free space for model weights

## Acknowledgements
We thank the Qwen, BLIP, and other open-source projects, as well as the IntentVC organizers, for their support.

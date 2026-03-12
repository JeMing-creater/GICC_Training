## requirements
```
pip install -r requirements.txt
```

run
```
CUDA_VISIBLE_DEVICES=0,2 accelerate launch main_dino.py
```

Tensorboard
```
tensorboard --logdir log/xxxxxxxxxx/colon_mri_dg_seg --port 6006 --host 0.0.0.0
```
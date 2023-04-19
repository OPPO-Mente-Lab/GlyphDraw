# GlyphDraw


[[Project Page](https://1073521013.github.io/glyph-draw.github.io/)] [[Paper](https://arxiv.org/abs/2303.17870)]


## Inference: Generate images with GlyphDraw

We provide one script to generate images using checkpoints. Include Clip checkpoints, GlyphDraw checkpoints. Then run
```bash
python test_en.py --clip_path=path_to_clip_checkpoints --model_id=path_to_GlyphDraw_checkpoints
```
In addition to the GlyphDraw series checkpoints, the checkpoints also requires a projection checkpoint for project image patch embedding, fusion checkpoint, and mask prediction checkpoint. If you want to predict masks instead of randomly assigning them.
One can check `test_en.py` for more details about interface. 
It should be noted that for Chinese models, an adaptive clip text encoder is required


## Training 

### Coming Soon
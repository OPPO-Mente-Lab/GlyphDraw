# GlyphDraw


[[Project Page](https://1073521013.github.io/glyph-draw.github.io/)] [[Paper](https://arxiv.org/abs/2303.17870)]


## Inference: Generate images with GlyphDraw

We provide one script to generate images using checkpoints. Include Clip checkpoints, GlyphDraw checkpoints. Then run
```bash
python test_en.py --clip_path=path_to_clip_checkpoints --model_id=path_to_GlyphDraw_checkpoints
```
In addition to the GlyphDraw series checkpoints, the checkpoints also requires a projection checkpoint for project image patch embedding, fusion checkpoint, and mask prediction checkpoint if you want to predict masks instead of randomly assigning them.
One can check `test_en.py` for more details about interface. 
It should be noted that is will require train clip text encoder additionally for Chinese models.


## Training 

### Coming Soon


## Acknowledgements 
This code is builds on the code from the [diffusers](https://github.com/huggingface/diffusers) library. 
[Chinese-CLIP](https://github.com/OFA-Sys/Chinese-CLIP) codebase.
[OpenCLIP](https://github.com/mlfoundations/open_clip) codebase.

# GlyphDraw


[[Project Page](https://1073521013.github.io/glyph-draw.github.io/)] [[Paper](https://arxiv.org/abs/2303.17870)]


## Requirements
A suitable [conda](https://conda.io/) environment named `glyphdraw` can be created
and activated with:

```
conda env create -f environment.yaml
conda activate glyphdraw
```

## Training 


```bash
bash train_en.sh 0 8
```
The first parameter represents The serial number of the current process, used for inter process communication. The host with rank=0 is the master node.
and the second parameter the world size.Please review the detailed parameters of model training
with train_en.sh script

## Inference: Generate images with GlyphDraw

We provide one script to generate images using checkpoints. Include Clip checkpoints, GlyphDraw checkpoints. Then run
```bash
python test_en.py --clip_path=path_to_clip_checkpoints --model_id=path_to_GlyphDraw_checkpoints
```
In addition to the GlyphDraw series checkpoints, the checkpoints also requires a projection checkpoint for project image patch embedding, fusion checkpoint, and mask prediction checkpoint. If you want to predict masks instead of randomly assigning them.
One can check `test_en.py` for more details about interface. 
It should be noted that for Chinese models, an adaptive clip text encoder is required


## TODOs

- [x] Release inference code
- [x] Release training code
- [ ] Release data preparation code
- [ ] Release mask prediction module training code
- [ ] Release demo


## Acknowledgements 
This code is builds on the code from the [diffusers](https://github.com/huggingface/diffusers) library. 
[Chinese-CLIP](https://github.com/OFA-Sys/Chinese-CLIP) codebase.
[OpenCLIP](https://github.com/mlfoundations/open_clip) codebase.
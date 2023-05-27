import os,re
import torch
import argparse
from pytorch_lightning import (
    LightningModule,
    Trainer,
)

from pytorch_lightning.callbacks import (
    LearningRateMonitor,
)
from model_utils import (
    add_module_args,
    configure_optimizers,
    get_total_steps,
)
from universal_checkpoint import UniversalCheckpoint
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from torch.nn import functional as F
from tqdm.auto import tqdm
from universal_datamodule_en import DataModuleCustom
from typing import Callable, List, Optional, Union
from einops import rearrange, repeat,reduce
from utils import load_config, load_clip, tokenize, save_config, save_model, numpy_to_pil, save_images
from torchvision.utils import save_image
import inspect
from cn_clip.clip import load_from_name
import cv2
import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import numpy as np
import random
import clip
import open_clip

from torchvision import transforms
from transformer import CausalTransformer

FONTS = "OPPOSans-S-B-0621.ttf"
RESUME_ID = False
RESUME_PATH = f"/public_data/ma/code/GLIGEN-master/results_mul/stablediffusion_glyphdraw_token_multi/"
UNET_CONFIG = {
    "act_fn": 'silu',
    "attention_head_dim": [
        5,
        10,
        20,
        20
    ],
    "block_out_channels": [
        320,
        640,
        1280,
        1280
    ],
    "center_input_sample": False,
    "cross_attention_dim": 1024,
    "down_block_types": [
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "DownBlock2D"
    ],
    "downsample_padding": 1,
    "dual_cross_attention": False,
    "flip_sin_to_cos": True,
    "freq_shift": 0,
    "in_channels": 6,
    "layers_per_block": 2,
    "mid_block_scale_factor": 1,
    "norm_eps": 1e-05,
    "norm_num_groups": 32,
    "num_class_embeds": None,
    "only_cross_attention": False,
    "out_channels": 4,
    "sample_size": 64,
    "up_block_types": [
        "UpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D"
    ],
    "use_linear_projection": True}

class StableDiffusion(LightningModule):
    @staticmethod
    def add_module_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('XXX Stable Diffusion Module')
        parser.add_argument('--train_unet', default=True)
        parser.add_argument('--train_clip_visual', default=False)
        parser.add_argument('--train_transformer', default=True)
        parser.add_argument('--local_rank', default=-1, type=int,help='node rank for distributed training')
        return parent_parser

    def __init__(self, args):
        super().__init__()
        self.text_encoder = StableDiffusionPipeline.from_pretrained(args.model_path)
        self.vae = AutoencoderKL.from_pretrained(args.model_path, subfolder="vae")
        self.unet = UNet2DConditionModel(**UNET_CONFIG)
        if RESUME_ID:
            self.unet.load_state_dict(torch.load(
                os.path.join(RESUME_PATH, f"unet_0_{RESUME_ID}/pytorch_model.bin")), strict=True)
        else:
            sd = torch.load(os.path.join(args.model_path,"unet/diffusion_pytorch_model.bin"))
            if "state_dict" in list(sd.keys()):
                sd = sd["state_dict"]
            keys = list(sd.keys())
            input_key="conv_in.weight"
            self_sd = self.unet.state_dict()
            input_weight =self_sd[input_key]
            input_weight.zero_()
            input_weight[:, :4, :, :].copy_(sd[input_key])
            del sd[input_key]

        self.test_scheduler = EulerDiscreteScheduler.from_pretrained(args.model_path, subfolder="scheduler")
        self.noise_scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
        self.save_hyperparameters(args)
        
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.RandomCrop(512),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        
        self.pattern = re.compile(r'\"(.*?)\"')
        self.model_clip, _, self.preprocess = open_clip.create_model_and_transforms("ViT-H-14", pretrained=args.clip_path)

        self.causal_transformer = CausalTransformer(dim = 1024)
        self.proj = torch.nn.Linear(1280, 1024)
        if RESUME_ID:
            self.causal_transformer.load_state_dict(torch.load(
                os.path.join(RESUME_PATH, f"transformer_0_{RESUME_ID}/pytorch_model.bin"), map_location="cpu"))
            self.proj.load_state_dict(torch.load(
                os.path.join(RESUME_PATH, f"proj_0_{RESUME_ID}/pytorch_model.bin"), map_location="cpu"))


    def setup(self, stage) -> None:
        if stage == 'fit':
            # 自己设置
            self.total_steps = 999999
            print('Total steps: {}' .format(self.total_steps))

    def configure_optimizers(self):
        model_params = []
        if self.hparams.train_unet:
            # model_params.append({'params': self.unet.conv_in.parameters()})
            # self.unet.down_blocks[0].attentions[0].transformer_blocks[0].attn2
            params=[]
            names = []
            for name, p in self.unet.named_parameters():
                # if ("transformer_blocks" in name) and  ("attn" in name):
                if ("transformer_blocks" in name) and  ("attn2" in name) and (("to_k" in name) or ("to_v" in name)):
                    params.append(p) 
                    names.append(name)
                elif "conv_in" in name:
                    params.append(p) 
                    names.append(name)
            total = sum(p.numel() for p in params)
            print(len(names),total)
            model_params.append({'params': iter(params)})

        if self.hparams.train_transformer:
            model_params.append({'params': self.causal_transformer.parameters()})
            model_params.append({'params': self.proj.parameters()})
        if self.hparams.train_clip_visual:
            model_params.append({'params': self.model_clip.parameters()})

        return configure_optimizers(self, model_params=model_params)

    def split_en(self, characters, words_len=22):
        characters_list = []
        tmp = ""
        remain = characters
        for i,c in enumerate(characters.split(" ")):
            if len(tmp+" "+c)<words_len:
                tmp = tmp+" "+c
            else:
                tmp = tmp.strip()
                characters_list.append(tmp)
                remain = remain.replace(tmp,"")
                tmp = c
        characters_list.append(remain.strip())
        return characters_list

    def font_devise_512(self,character,single_channal):
        width_x = 512
        width_y = 512
        if single_channal:
            img=Image.new("L", (width_x,width_y),255)
        else:
            img=Image.new("RGB", (width_x,width_y),(255,255,255))

        if not character:
            return img

        # 字体大小
        if len(character)<10:
            character_size = 80
        elif len(character)<15:
            character_size = 55
        else:
            character_size = 45

        # 判断是否分行
        character_list = self.split_en(character)

        font = ImageFont.truetype(FONTS,character_size)
        chars_w, chars_h = font.getsize(character_list[0])
        draw = ImageDraw.Draw(img)
        chars_y = int((width_y - chars_h)/2)
        # 横图
        if len(character_list)==1:
            chars_x = int((width_x - chars_w)/2)
            draw.text((chars_x,chars_y),character_list[0],fill="Black",font=font)
        # 竖图
        else:
            chars_y_0 = chars_y - int((len(character_list))/2)*character_size
            for j in range(len(character_list)):
                character = character_list[j]
                chars_x = 40
                if j==0:
                    draw.text((chars_x,chars_y_0),character,fill="Black",font=font)
                    chars_y = chars_y_0 + (character_size+10)
                else:
                    draw.text((chars_x,chars_y),character,fill="Black",font=font)
                    chars_y += (character_size+10)
        return img

    def font_devise_64(self,character,single_channal):
        width_x = 64
        width_y = 64
        if single_channal:
            img=Image.new("L", (width_x,width_y),255)
        else:
            img=Image.new("RGB", (width_x,width_y),(255,255,255))
        if not character:
            return img
        if len(character)<6:
            character_size = 12
        elif len(character)<10:
            character_size = 10
        else:
            character_size = 8

        # 判断是否分行
        character_list = self.split_en(character,13)

        font = ImageFont.truetype(FONTS,character_size)
        chars_w, chars_h = font.getsize(character_list[0])
        draw = ImageDraw.Draw(img)
        chars_y = int((width_y - chars_h)/2)
        # 横图
        if len(character_list)==1:
            chars_x = int((width_x - chars_w)/2)
            draw.text((chars_x,chars_y),character_list[0],fill="Black",font=font)
        # 竖图
        else:
            chars_y_0 = chars_y - int((len(character_list))/2)*character_size
            for j in range(len(character_list)):
                character = character_list[j]
                chars_x = 5
                if j==0:
                    draw.text((chars_x,chars_y_0),character,fill="Black",font=font)
                    chars_y = chars_y_0 + (character_size)
                else:
                    draw.text((chars_x,chars_y),character,fill="Black",font=font)
                    chars_y += (character_size)
        return img

    def encode_images(self, prompt,masks, device):
        image_tensor = []
        for character in prompt:
            img = self.font_devise_512(character,False)
            image_tensor.append(self.preprocess(img).unsqueeze(0).to(device))
        with torch.no_grad():
            image_embeddings = self.model_clip.to(device).float().encode_image(torch.cat(image_tensor))
            image_embeddings = self.proj(image_embeddings[1].half())
        return image_embeddings

    def encode_images_vae(self, prompt,masks, device):
        image_tensor = []
        for character,mask in zip(prompt,masks):
            img = self.font_devise_512(character,True)
            image_tensor.append(self.image_transforms(img).unsqueeze(0).to(device))
        pixel_values = torch.cat(image_tensor)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float().half()
        latents =  torch.nn.functional.interpolate(pixel_values, size=(64, 64))
        return latents

    # 测试用
    def encode_images_mask(self, prompt, device):
        width_x = 512
        width_y = 512
        image_tensor, masks = [], []
        for character in prompt:
            character_size = 45
            # assert len(character)<13
            character = character[:12]
            font = ImageFont.truetype(FONTS,character_size)
            chars_w, chars_h = font.getsize(character)
            seed = random.uniform(0.6,1) # 字体大小一定比例缩小
            chars_w, chars_h = int(chars_w*seed),int(chars_h*seed)
            chars_x = random.randint(0, int((width_x - chars_w)))
            chars_y = random.randint(0, int((width_y - chars_h)))
            mask_img = np.zeros((width_x, width_y))
            mask_img[chars_y: chars_y + chars_h, chars_x: chars_x + chars_w] = 1
            mask_img = Image.fromarray(mask_img)
            mask_img_resize = transforms.Resize((64, 64), interpolation=transforms.InterpolationMode.NEAREST)(mask_img)
            mask_tensor_resize = 1 - transforms.ToTensor()(mask_img_resize)
            masks.append(mask_tensor_resize.unsqueeze(1))
        
        return torch.cat(masks).to(device)


    @torch.no_grad()
    def encode_prompt(self, prompt, font,latents_c2,device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt):
        batch_size = len(prompt) if isinstance(prompt, list) else 1
        self.text_encoder = self.text_encoder.to(device)
        text_embeddings = self.text_encoder._encode_prompt(prompt,device=device,num_images_per_prompt=1,do_classifier_free_guidance=False)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            uncond_embeddings = self.text_encoder._encode_prompt(uncond_tokens,device=device,num_images_per_prompt=1,do_classifier_free_guidance=False)

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_images_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_images_per_prompt, seq_len, -1)

            image_embedding = self.encode_images(font,latents_c2,device) # batch*1024
            # image_embedding = image_embedding.unsqueeze(1)
            # image_embedding = torch.nn.functional.pad(image_embedding, pad=(0, 0, 0, 63), mode='replicate', value=0) # batch*64*1024
            co_embedding = torch.cat([text_embeddings,image_embedding],1)
            # co_embedding = reduce(co_embedding, '(b h) n d -> b n d', 'mean', h=2)
            co_embedding_convert = self.causal_transformer(co_embedding.half())
            uncond_embeddings = torch.nn.functional.pad(uncond_embeddings, pad=(0, 0, 0, 256), mode='constant', value=0) # batch*65*1024
            text_embeddings = torch.cat([co_embedding_convert, uncond_embeddings, uncond_embeddings])
      
        return text_embeddings
    
    def prepare_extra_step_kwargs(self, generator, eta):
        accepts_eta = "eta" in set(inspect.signature(self.test_scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.test_scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    @torch.no_grad()
    def log_imgs(
        self,
        device,
        prompts: Union[str, List[str]],
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        num_images_per_prompt: Optional[int] = 1,
        **kwargs,
    ):
        batch_size = 1 if isinstance(prompts, str) else len(prompts)
        do_classifier_free_guidance = guidance_scale > 1.0
        
        font = []
        prompt = []
        for t in prompts:
            font_sin = self.pattern.findall(t)
            if font_sin:
                font_sin = font_sin[0]
            else:
                font_sin = ""
            font.append(font_sin)
            prompt.append(t.replace(font_sin,"").replace("\"",""))

        latents_c2 = self.encode_images_mask(font,device)
        text_embeddings = self.encode_prompt(prompt, font,latents_c2,device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt)

        self.test_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.test_scheduler.timesteps

        shape = (batch_size * num_images_per_prompt, 4, height // 8, width // 8)

        latents = torch.randn(shape, generator=generator, device=device, dtype=text_embeddings.dtype)
        latents = latents * self.test_scheduler.init_noise_sigma

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        latents_c1 = self.encode_images_vae(font,latents_c2,device)
        font_latents = torch.cat([latents_c1,latents_c2],1)
        uncond_image_latents = torch.zeros_like(font_latents).to(device)
        font_latents = torch.cat([font_latents, font_latents, uncond_image_latents], dim=0).to(device).half()

        for i, t in enumerate(tqdm(timesteps)):

            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 3) if do_classifier_free_guidance else latents
            latent_model_input = self.test_scheduler.scale_model_input(latent_model_input, t)
            latent_model_input = torch.cat([latent_model_input, font_latents], dim=1)

            noise_pred = self.unet(latent_model_input.half(), t, encoder_hidden_states=text_embeddings.half()).sample
            if do_classifier_free_guidance:
                noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
                noise_pred = (
                            noise_pred_uncond
                            + guidance_scale * (noise_pred_text - noise_pred_image)
                            + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
                        )
                

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.test_scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents.half()).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = numpy_to_pil(image)

        return image

    def training_step(self, batch, batch_idx):
        if self.hparams.train_unet:
            for name, p in self.unet.named_parameters():
                if ("transformer_blocks" in name) and  ("attn2" in name) and (("to_k" in name) or ("to_v" in name)):
                    p.requires_grad_(True)
                elif "conv_in" in name:
                    p.requires_grad_(True)
                else:
                    p.requires_grad_(False)

        if self.hparams.train_clip_visual:
            self.model_clip.train()
        else:
            self.model_clip.requires_grad_(False)

        if self.hparams.train_transformer:
            self.causal_transformer.train()
            self.proj.train()
        else:
            self.causal_transformer.requires_grad_(False)

        with torch.no_grad():
            latents = self.vae.encode(batch["pixel_values"]).latent_dist.sample()
            latents = latents * 0.18215

        # Sample noise that we'll add to the latents
        noise = torch.randn(latents.shape).to(latents.device)
        noise = noise.to(dtype=self.unet.dtype)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()
        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)


        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        noisy_latents = noisy_latents.to(dtype=self.unet.dtype)

        mask_font = batch["mask"]

        # Get the text embedding for conditioning
        self.text_encoder = self.text_encoder.to(latents.device)
        encoder_hidden_states = self.text_encoder._encode_prompt(batch["texts"],device=latents.device,num_images_per_prompt=1,do_classifier_free_guidance=False)

        # Predict the noise residual
        latents_c1 = self.encode_images_vae(batch["font"],batch["mask"],latents.device)
        latents_c2 = mask_font
        latents_c = torch.cat([latents_c1,latents_c2],1)
        # To support classifier-free guidance, randomly drop out only text conditioning 5%, only image conditioning 5%, and both 5%.
        uncond=0.05
        uncond_image = 0.05
        random = torch.rand(latents.size(0), device=latents.device)
        prompt_mask = rearrange(random < 2 * uncond, "n -> n 1 1")
        input_mask = 1 - rearrange((random >= uncond_image).float() * (random < 3 * uncond_image).float(), "n -> n 1 1 1")
        # input_mask = 1 - rearrange((random >= uncond).float() * (random < 3 * uncond).float(), "n -> n 1 1 1")
        null_prompt = self.text_encoder._encode_prompt(len(batch["texts"])*[""],device=latents.device,num_images_per_prompt=1,do_classifier_free_guidance=False)
        encoder_hidden_states = torch.where(prompt_mask, null_prompt, encoder_hidden_states)

        image_embedding = self.encode_images(batch["font"],batch["mask"],latents.device) # batch*256*1024
        encoder_hidden_states = torch.cat([encoder_hidden_states,image_embedding],1)
        encoder_hidden_states = self.causal_transformer(encoder_hidden_states.half())

        noise_pred = self.unet(torch.cat([noisy_latents.half(), input_mask.half()*latents_c.half()],1), timesteps, encoder_hidden_states).sample

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        loss_weights = (1 - batch['mask']) * 0.6 + torch.ones_like(batch['mask'])
        loss = F.mse_loss(noise_pred, noise, reduction="none")*loss_weights
        loss = loss.mean([1, 2, 3]).mean()
        self.log("train_loss", loss.item(),  on_epoch=False, prog_bar=True, logger=True)
        self.log("lr", lr,  on_epoch=False, prog_bar=True, logger=True)

        if self.trainer.global_rank == 0:
            if (self.global_step+1) % 4000 == 0:
                print('saving model...')
                save_path = os.path.join(args.default_root_dir, f'hf_out_{self.trainer.current_epoch}_{self.global_step}')
                save_path_unet = os.path.join(args.default_root_dir, f'unet_{self.trainer.current_epoch}_{self.global_step}')
                if self.hparams.train_unet:
                    save_model(self.unet, save_path_unet)
                save_model(self.proj, os.path.join(args.default_root_dir, f'proj_{self.trainer.current_epoch}_{self.global_step}'))
                save_model(self.causal_transformer, os.path.join(args.default_root_dir, f'transformer_{self.trainer.current_epoch}_{self.global_step}'))
                if self.hparams.train_clip_visual:
                        save_model(self.model_clip, os.path.join(args.default_root_dir, f'clip_{self.trainer.current_epoch}_{self.global_step}'))
                # 生成测试图片
                with torch.no_grad():
                    try:
                        with open(args.test_prompts, "r", encoding='utf-8') as f:
                            prompts = f.readlines()[:6]
                        prompts = [line.strip() for line in prompts]
                        assert prompts
                    except Exception:
                        print(f"No prompts read from file: {args.test_prompts}, skip test.")
                    else:
                        print(prompts)
                        images = self.log_imgs(latents.device, prompts, num_images_per_prompt=args.test_repeat)
                        img_path = os.path.join(save_path, "test_images")
                        save_images(images, img_path, prompts, args.test_repeat)
                        print("Test images saved to: {}".format(img_path))

        return {"loss": loss}

    def on_load_checkpoint(self, checkpoint) -> None:
        # 兼容低版本lightning，低版本lightning从ckpt起来时steps数会被重置为0
        global_step_offset = checkpoint["global_step"]
        if 'global_samples' in checkpoint:
            self.consumed_samples = checkpoint['global_samples']
        self.trainer.fit_loop.epoch_loop._batches_that_stepped = global_step_offset


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser = add_module_args(args_parser)
    args_parser = DataModuleCustom.add_data_specific_args(args_parser)
    args_parser = Trainer.add_argparse_args(args_parser)
    args_parser = StableDiffusion.add_module_specific_args(args_parser)
    args_parser = UniversalCheckpoint.add_argparse_args(args_parser)
    args = args_parser.parse_args()

    model = StableDiffusion(args)
    
    def collate_fn(examples):
        # print(examples)
        texts = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]

        font = [example["font"] for example in examples]
        mask = torch.stack([example["mask"] for example in examples])
        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        batch = {
            "font": font,
            "texts": texts,
            "pixel_values": pixel_values,
            "mask":mask
        }

        return batch

    datamoule = DataModuleCustom(args, tokenizer=None, collate_fn=collate_fn)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = UniversalCheckpoint(args)

    trainer = Trainer.from_argparse_args(args,
                                         callbacks=[
                                             lr_monitor,
                                             checkpoint_callback])

    trainer.fit(model, datamoule, ckpt_path=args.load_ckpt_path)

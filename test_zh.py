import os,re
import torch
import argparse

import random
from utils import load_config, load_clip, tokenize

from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler,DPMSolverMultistepScheduler
from typing import Callable, List, Optional, Union

import torch.nn as nn
import torch.optim as optim

from cn_clip.clip import FullTokenizer
from cn_clip.clip import load_from_name

import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import numpy as np

from torchvision.utils import save_image
from transformer import CausalTransformer
from torchvision import transforms
import math

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

class latent_guidance_predictor(nn.Module):
    def __init__(self, output_dim, input_dim, num_encodings):
        super(latent_guidance_predictor, self).__init__()
        self.num_encodings = num_encodings
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=512),
            nn.Linear(512, 256),         
            nn.ReLU(),
            nn.BatchNorm1d(num_features=256),
            nn.Linear(256, 128),     
            nn.ReLU(),
            nn.BatchNorm1d(num_features=128),
            nn.Linear(128, 64),      
            nn.ReLU(),
            nn.BatchNorm1d(num_features=64),
            nn.Linear(64, output_dim)
        )

    def forward(self, x, t):
        # Concatenate input pixels with noise level t and positional encodings
        pos_encoding = [torch.sin(2 * math.pi * t * (2 **-l)) for l in range(self.num_encodings)]
        pos_encoding = torch.cat(pos_encoding, dim=-1)
        x = torch.cat((x, t, pos_encoding), dim=-1)
        x = x.flatten(start_dim=0, end_dim=3)
        
        return self.layers(x)

def save_tensors(module: nn.Module, features, name: str):
    """ Process and save activations in the module. """
    if type(features) in [list, tuple]:
        features = [f.detach().float() for f in features if f is not None and isinstance(f, torch.Tensor)]
        setattr(module, name, features)
    elif isinstance(features, dict):
        features = {k: f.detach().float() for k, f in features.items()}
        setattr(module, name, features)
    else:
        setattr(module, name, features.detach().float())

def save_out_hook(self, inp, out):
    save_tensors(self, out, 'activations')
    return out

def resize_and_concatenate(activations: List[torch.Tensor], reference):
    assert all([isinstance(acts, torch.Tensor) for acts in activations])
    size = reference.shape[2:]
    resized_activations = []
    for acts in activations:
        acts = nn.functional.interpolate(
            acts, size=size, mode="bilinear"
        )
        acts = acts[:1]
        acts = acts.transpose(1,3) # b*64*64*320
        resized_activations.append(acts)
    
    return torch.cat(resized_activations, dim=3)

class StableDiffusionTest():

    def __init__(self, clip_cn_adapter, model_id,transformer_id,proj_id,mpm_id,clip_path,device,random_mask,fonts):
        super().__init__()
        self.tokenizer = FullTokenizer()
        self.bert_config = load_config(clip_cn_adapter)
        self.text_encoder = load_clip(clip_cn_adapter, self.bert_config).to(device).eval()

        scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler,torch_dtype=torch.float16).to(device)
        self.model_clip, self.preprocess = load_from_name(clip_path, device=device)
        self.causal_transformer = CausalTransformer(dim = 1024).to(device)
        s = torch.load(transformer_id, map_location="cpu")
        self.causal_transformer.load_state_dict(s)
        self.proj = torch.nn.Linear(1280, 1024).to(device)
        self.proj.load_state_dict(torch.load(proj_id, map_location="cpu"))
        
        self.pattern = re.compile(r'“(.*?)”')
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.RandomCrop(512),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.image_transforms_mask = transforms.Compose(
            [
                transforms.Resize(64, interpolation=transforms.functional._interpolation_modes_from_int(0)),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
            ]
        )

        self.model_mpm = latent_guidance_predictor(output_dim=4, input_dim=7080, num_encodings=9).to(device)
        checkpoint = torch.load(mpm_id, map_location="cpu")
        self.model_mpm.load_state_dict(checkpoint)
        self.model_mpm.eval()
        save_hook = save_out_hook
        blocks = [0,1,2,3]
        self.feature_blocks = []
        for idx, block in enumerate(self.pipe.unet.down_blocks):
            if idx in blocks:
                block.register_forward_hook(save_hook)
                self.feature_blocks.append(block) 
                
        for idx, block in enumerate(self.pipe.unet.up_blocks):
            if idx in blocks:
                block.register_forward_hook(save_hook)
                self.feature_blocks.append(block)  
        self.random_mask = random_mask
        self.fonts = fonts
        self.device = device

    def font_devise(self,character,single_channal):
        character = re.sub('[^\u4e00-\u9fa5]+', '', character)
        width_x = 512
        width_y = 512
        if single_channal:
            img=Image.new("L", (width_x,width_y),255)
        else:
            img=Image.new("RGB", (width_x,width_y),(255,255,255))
        if len(character)<8:
            character_size = 200-len(character)*20
            font = ImageFont.truetype(self.fonts,character_size)
            _,_,chars_w, chars_h = font.getbbox(character)
            chars_x = int((width_x - chars_w)/2)
            chars_y = int((width_y - chars_h)/2)
            draw = ImageDraw.Draw(img)
            draw.text((chars_x,chars_y),character,fill="Black",font=font)

        else:
            character_size = 70
            font = ImageFont.truetype(self.fonts,character_size)
            _,_,chars_w, chars_h = font.getbbox(character[:7])
            chars_x = int((width_x - chars_w)/2)
            chars_y = int((width_y - chars_h)/2)
            draw = ImageDraw.Draw(img)
            chars_y -= int((int(len(character)/7)+1)/2)*character_size
            for i in range(int(len(character)/7)+1):
                draw.text((chars_x,chars_y),character[i*7:(i+1)*7],fill="Black",font=font)
                chars_y += character_size
        return img
    

    def encode_images(self, prompt, device):
        width_x = 512
        width_y = 512
        image_tensor = []
        for character in prompt:
            img = self.font_devise(character,False)
            image_tensor.append(self.preprocess(img).unsqueeze(0).to(device))
        with torch.no_grad():
            image_embeddings = self.model_clip.to(device).float().encode_image_tokens(torch.cat(image_tensor))
            image_embeddings = self.proj(image_embeddings)
            # image_embeddings = self.model_clip.to(device).float().encode_image(torch.cat(image_tensor))
        return image_embeddings


    def encode_images_vae(self, prompt, device):
        width_x = 512
        width_y = 512
        vae_scale_factor = 8
        image_tensor = []
        for character in prompt:
            img = self.font_devise(character,True)
            image_tensor.append(self.image_transforms(img).unsqueeze(0).to(device))

        pixel_values = torch.cat(image_tensor)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float().half()
        latents =  torch.nn.functional.interpolate(pixel_values, size=(width_x // vae_scale_factor, width_y // vae_scale_factor))

        return latents

    def encode_images_mask(self, prompt, device):
        width_x = 512
        width_y = 512
        image_tensor, masks = [], []
        for character in prompt:
            seed = random.uniform(0.3,1) # 字体大小一定比例缩小
            if len(character)<8:
                character_size = 200-len(character)*20
                font = ImageFont.truetype(self.fonts,character_size)
                _,_,chars_w, chars_h = font.getbbox(character)
                
                chars_w, chars_h = int(chars_w*seed),int(chars_h*seed)
                chars_x = random.randint(0, int((width_x - chars_w)))
                chars_y = random.randint(0, int((width_y - chars_h)))
            else:
                character_size = 70
                font = ImageFont.truetype(self.fonts,character_size)
                _,_,chars_w, chars_h = font.getbbox(character[:7])
                chars_w, chars_h = int(chars_w*seed),int(chars_h*seed)
                chars_x = random.randint(0, int((width_x - chars_w)))
                chars_y = random.randint(0, int((width_y - chars_h)))
                chars_y -= int((int(len(character)/7)+1)/2)*character_size
                for i in range(int(len(character)/7)+1):
                    chars_y += character_size

            mask_img = np.zeros((width_x, width_y))
            if random.random()>0.5:
                mask_img[chars_y: chars_y + chars_h, chars_x: chars_x + chars_w] = 1
            else:
                mask_img[chars_x: chars_x + chars_w, chars_y: chars_y + chars_h] = 1
            # mask_img[chars_y: chars_y + chars_h, chars_x: chars_x + chars_w] = 1
            mask_img = Image.fromarray(mask_img)
            mask_img_resize = transforms.Resize((64, 64), interpolation=transforms.InterpolationMode.NEAREST)(mask_img)
            mask_tensor_resize = 1 - transforms.ToTensor()(mask_img_resize)
            masks.append(mask_tensor_resize.unsqueeze(1))
        
        return torch.cat(masks).to(device)

    def encode_prompt(self, prompts,font, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt):
        batch_size = len(prompts) if isinstance(prompts, list) else 1
        text_input_ids = tokenize(self.tokenizer, prompts)
        pad_index = self.tokenizer.vocab['[PAD]']
        attention_mask = text_input_ids.ne(pad_index).type(self.text_encoder.embeddings.word_embeddings.weight.dtype).to(device)

        text_embeddings = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        text_embeddings = text_embeddings[0]

        # duplicate text embeddings for each generation per prompts, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompts) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompts`, but got {type(negative_prompt)} !="
                    f" {type(prompts)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompts`:"
                    f" {prompts} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompts`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input_ids = tokenize(self.tokenizer, uncond_tokens)
            pad_index = self.tokenizer.vocab['[PAD]']
            uncond_attention_mask = uncond_input_ids.ne(pad_index).type(self.text_encoder.embeddings.word_embeddings.weight.dtype).to(device)

            uncond_embeddings = self.text_encoder(
                # uncond_input.input_ids.to(device),
                uncond_input_ids.to(device),
                attention_mask=uncond_attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for each generation per prompts, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_images_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_images_per_prompt, seq_len, -1)

            image_embedding = self.encode_images(font,device) # batch*1024
            co_embedding = torch.cat([text_embeddings,image_embedding.half()],1)
            co_embedding_convert = self.causal_transformer(co_embedding)
            uncond_embeddings = torch.nn.functional.pad(uncond_embeddings, pad=(0, 0, 0, 257), mode='constant', value=0) # batch*65*1024
            text_embeddings = torch.cat([co_embedding_convert, uncond_embeddings, uncond_embeddings])
         
        return text_embeddings

    @torch.no_grad()
    def __call__(
        self,
        prompts: Union[str, List[str]],
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        **kwargs,
    ):
        # 0. Default height and width to unet
        height = height or self.pipe.unet.config.sample_size * self.pipe.vae_scale_factor
        width = width or self.pipe.unet.config.sample_size * self.pipe.vae_scale_factor

        font = []
        prompt = []
        for t in prompts:
            font_sin = self.pattern.findall(t)
            if font_sin:
                font_sin = font_sin[0]
            else:
                font_sin = ""
            font.append(font_sin)
            prompt.append(t.replace(font_sin,"").replace("“","").replace("”",""))


        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self.pipe._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        text_embeddings = self.encode_prompt(prompt,font, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt).half()

        # 4. Prepare timesteps
        self.pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.pipe.scheduler.timesteps

        # 5. Prepare latent variables
        # 
        num_channels_latents = self.pipe.unet.in_channels-2
        latents = self.pipe.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            latents,
        ).half()

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.pipe.prepare_extra_step_kwargs(generator, eta)


        latents_c1 = self.encode_images_vae(font,device)
        if not self.random_mask:
            latents_c2 = torch.zeros_like(latents_c1).to(device)
        else:
            latents_c2 = self.encode_images_mask(font,device)
        font_latents = torch.cat([latents_c1,latents_c2],1)


        uncond_image_latents = torch.zeros_like(font_latents).to(device)
        font_latents = torch.cat([font_latents, font_latents, uncond_image_latents], dim=0)
        latents_copy = latents.clone()

        if not self.random_mask:
            # 7. Denoising loop
            for i, t in enumerate(self.pipe.progress_bar(timesteps)):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 3) if do_classifier_free_guidance else latents
                latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t)
                latent_model_input = torch.cat([latent_model_input, font_latents], dim=1)
                # predict the noise residual
                noise_pred = self.pipe.unet(latent_model_input.half(), t, encoder_hidden_states=text_embeddings).sample

                # perform guidance
                if do_classifier_free_guidance:
                    # noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
                    noise_pred = (
                                noise_pred_uncond
                                + guidance_scale * (noise_pred_text - noise_pred_image)
                                + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
                            )
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                ##### predict mpm, here means predict font mask 
                if i==20:
                    features, encoded_edge_maps, noise_levels = [], [], []
                    latents_c1s = []
                    for ii in range(batch_size):
                        noisy_latent = latents[ii].unsqueeze(0)  # 训练中就是加噪的真实图片，预测就是predict中间输出
                        latent_model_input = torch.cat([noisy_latent] * 3)
                        latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t)
                        latents_c1 = self.encode_images_vae([font[ii]],device)
                        latents_c2 = torch.zeros_like(latents_c1).to(device)
                        font_latents_vae = torch.cat([latents_c1,latents_c2],1)
                        uncond_image_latents = torch.zeros_like(font_latents_vae).to(device)
                        font_latents_tmp = torch.cat([font_latents_vae, font_latents_vae, uncond_image_latents], dim=0)
                        latent_model_input = torch.cat([latent_model_input, font_latents_tmp], dim=1)
                        
                        activations = []
                        with torch.no_grad():
                            text_embeddings_1 = self.encode_prompt([prompt[ii]],[font[ii]], device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt).half()
                            _ = self.pipe.unet(latent_model_input.half(), t, encoder_hidden_states=text_embeddings_1).sample

                        # Extract activations
                        for block in self.feature_blocks:
                            activations.append(block.activations)
                            block.activations = None
                            
                        activations = [activations[0][0], activations[1][0], activations[2][0], activations[3][0], activations[4], activations[5], activations[6], activations[7]]
                        
                        feature =  resize_and_concatenate(activations, noisy_latent)
                        # sqrt_alpha_prod = self.test_scheduler.alphas_cumprod[timesteps].to(latents.device) ** 0.5
                        # sqrt_alpha_prod = sqrt_alpha_prod.flatten()
                        # while len(sqrt_alpha_prod.shape) < len(noisy_latent.shape):
                        #     sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
                        sqrt_one_minus_alpha_prod = (1 - self.pipe.scheduler.alphas_cumprod[t]).to(latents.device) ** 0.5
                        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
                        while len(sqrt_one_minus_alpha_prod.shape) < len(noisy_latent.shape):
                            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
                        # noise_level = noisy_latent - noise_pred[ii].unsqueeze(0)  ## 这里x_t-1 - noise_pred  ?
                        noise_level = noise_pred[ii].unsqueeze(0)*sqrt_one_minus_alpha_prod
                        noise_level =  noise_level.transpose(1,3)
                        features.append(feature.unsqueeze(0))
                        noise_levels.append(noise_level.unsqueeze(0))
                        latents_c1s.append(latents_c1)
                    features = torch.cat(features)
                    noise_levels = torch.cat(noise_levels)
                    latents_c1s = torch.cat(latents_c1s)

                    pred_edge_map = self.model_mpm(features, noise_levels).unflatten(0, (batch_size, 64, 64)).transpose(3, 1)
                    pred_edge = torch.gt(pred_edge_map, 0.5).long()
                    pred_edge = pred_edge[:,0].unsqueeze(1)

                    font_latents_lgb = torch.cat([latents_c1s,pred_edge],1)
                    uncond_image_latents = torch.zeros_like(font_latents_lgb).to(device)
                    font_latents = torch.cat([font_latents_lgb, font_latents_lgb, uncond_image_latents], dim=0)
                    break


        # 7. Denoising loop
        for i, t in enumerate(self.pipe.progress_bar(timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents_copy] * 3) if do_classifier_free_guidance else latents_copy
            latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t)
            latent_model_input = torch.cat([latent_model_input, font_latents], dim=1)
            # predict the noise residual
            noise_pred = self.pipe.unet(latent_model_input.half(), t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            if do_classifier_free_guidance:
                # noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
                noise_pred = (
                            noise_pred_uncond
                            + guidance_scale * (noise_pred_text - noise_pred_image)
                            + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
                        )
            # compute the previous noisy sample x_t -> x_t-1
            latents_copy = self.pipe.scheduler.step(noise_pred, t, latents_copy, **extra_step_kwargs).prev_sample


        # 8. Post-processing
        image = self.pipe.decode_latents(latents_copy)

        # # 9. Run safety checker
        # image, has_nsfw_concept = self.pipe.run_safety_checker(image, device, text_embeddings.dtype)

        # 10. Convert to PIL
        if output_type == "pil":
            image = self.pipe.numpy_to_pil(image)

        return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,  default="checkpoints/GlyphDraw_zh", help="GlyphDraw model folder")
    parser.add_argument("--clip_path", type=str,  default="checkpoints/clip_cn_vit-h-14.pt", help="clip model folder")
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--fonts", type=str, default="OPPOSans-S-B-0621.ttf", help="System font generation, which needs to be placed in the specified system file")
    parser.add_argument("--device", type=str,  default="cuda")
    parser.add_argument("--random_mask", type=bool,  default=False, help="Is the mask given randomly during inference")
    args = parser.parse_args()
    
    # ids = "19999"
    # files = "results_mul/stablediffusion_new"
    # model_id = "/public_data/ma/stable_models/model_font_6"  # model_font_6
    # font_unet_bin = f"/public_data/ma/code/stablediffusion-controlnet/{files}/unet_0_{ids}/pytorch_model.bin"
    # os.system("cp {} {}".format(font_unet_bin,os.path.join(model_id,"unet/diffusion_pytorch_model.bin")))
    # clip_cn_adapter = f"/public_data/ma/code/stablediffusion-controlnet/{files}/hf_out_0_{ids}/"
    # transformer_id = f"/public_data/ma/code/stablediffusion-controlnet/{files}/transformer_0_{ids}/pytorch_model.bin"
    # proj_id = f"/public_data/ma/code/stablediffusion-controlnet/{files}/proj_0_{ids}/pytorch_model.bin"
    # clip_cn = "/public_data/ma/models/clip_cn_vit-h-14.pt"
    # mpm_id = "/public_data/ma/code/stablediffusion-font/result/stablediffusion_mpm_zh/mpm_0_9999/pytorch_model.bin"

    model_id = args.model_path
    clip_cn_adapter = os.path.join(model_id,"text_encoder/pytorch_model.bin")
    transformer_id = os.path.join(model_id,"transformer_id.bin")
    proj_id = os.path.join(model_id,"projection_id.bin")
    mpm_id = os.path.join(model_id,"mpm_id.bin")
    sdt = StableDiffusionTest(clip_cn_adapter, model_id,transformer_id,proj_id,mpm_id,args.clip_path,args.device,args.random_mask,args.fonts)

    while True:
        raw_text = input("\nPlease Input Query (stop to exit) >>> ")
        if not raw_text:
            print('Query should not be empty!')
            continue
        if raw_text == "stop":
            break
        images = sdt([raw_text]*args.batch_size)
        grid = image_grid(images, rows=1, cols=args.batch_size)
        grid.save("glyphdraw.png")


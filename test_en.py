import os,re,sys
import argparse
import math
from typing import Callable, List, Optional, Union

import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import cv2
import numpy as np
import torch.nn as nn
import torch
from transformer import CausalTransformer
from torchvision import transforms
import open_clip
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler,DPMSolverMultistepScheduler


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

    def __init__(self, clip_path, model_id,transformer_id,proj_id,mpm_id,device,random_mask,fonts):
        super().__init__()
        self.text_encoder = StableDiffusionPipeline.from_pretrained(model_id)
        self.lr = 0.0001
        self.T = 20
        scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler,torch_dtype=torch.float16).to(device) # , revision="fp16", torch_dtype=torch.float16
        self.model_clip, _, self.preprocess = open_clip.create_model_and_transforms("ViT-H-14", pretrained=clip_path)
     
        self.causal_transformer = CausalTransformer(dim = 1024).to(device)
        self.causal_transformer.load_state_dict(torch.load(transformer_id, map_location="cpu"))
        self.proj = torch.nn.Linear(1280, 1024).to(device)
        self.proj.load_state_dict(torch.load(proj_id, map_location="cpu"))
        
        self.pattern = re.compile(r'\"(.*?)\"')
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.RandomCrop(512),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
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

    def split_en(self, characters, words_len=13):
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

    def add_random(self,pred_edge,prompt,device):
        result_new = []
        for pred_mask,p in zip(pred_edge,prompt):
            pixels = len(torch.where(pred_mask.squeeze()==0)[0]) #total 4096
            if (pixels<200 or pixels>3000) and p:
                result_new.append(self.encode_images_mask(p,device).unsqueeze(0))
            else:
                result_new.append(pred_mask.unsqueeze(0))

        return torch.cat(result_new)
        
    def font_devise_64(self,character,single_channal):
        width_x = 64
        width_y = 64
        if single_channal:
            img=Image.new("L", (width_x,width_y),255)
        else:
            img=Image.new("RGB", (width_x,width_y),(255,255,255))
        if not character:
            return img,None
        if len(character)<6:
            character_size = 12
        elif len(character)<10:
            character_size = 10
        else:
            character_size = 8

        # 判断是否分行
        character_list = self.split_en(character)

        font = ImageFont.truetype(self.fonts,character_size)
        _,_,chars_w, chars_h = font.getbbox(character_list[0])
        draw = ImageDraw.Draw(img)
        chars_y = int((width_y - chars_h)/2)
        # 横图
        if len(character_list)==1:
            chars_x = int((width_x - chars_w)/2)
            draw.text((chars_x,chars_y),character_list[0],fill="Black",font=font)
            return img,(chars_x,chars_y,chars_w,chars_h)
        # 竖图
        else:
            chars_y_0 = chars_y - int((len(character_list))/2)*character_size
            for j in range(len(character_list)):
                character = character_list[j]
                chars_x = 3
                if j==0:
                    draw.text((chars_x,chars_y_0),character,fill="Black",font=font)
                    chars_y = chars_y_0 + (character_size)
                else:
                    draw.text((chars_x,chars_y),character,fill="Black",font=font)
                    chars_y += (character_size)
            return img,(chars_x,chars_y_0,chars_w,int((len(character_list))/2)*character_size)
     
    def font_devise_512(self,character,single_channal):
        width_x = 512
        width_y = 512
        if single_channal:
            img=Image.new("L", (width_x,width_y),255)
        else:
            img=Image.new("RGB", (width_x,width_y),(255,255,255))
        if not character:
            return img
        if len(character)<10:
            character_size = 80
        elif len(character)<15:
            character_size = 55
        else:
            character_size = 45

        # 判断是否分行
        character_list = self.split_en(character,22)

        font = ImageFont.truetype(self.fonts,character_size)
        _,_,chars_w, chars_h = font.getbbox(character_list[0])
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

    def encode_images(self, prompt, device):
        image_tensor = []
        for character in prompt:
            img = self.font_devise_512(character,False)
            image_tensor.append(self.preprocess(img).unsqueeze(0).to(device))
        with torch.no_grad():
            image_embeddings = self.model_clip.to(device).float().encode_image(torch.cat(image_tensor))
            image_embeddings = self.proj(image_embeddings[1])
        return image_embeddings


    def encode_images_vae(self, prompt, device):
        image_tensor = []
        for character in prompt:
            img = self.font_devise_512(character,True)
            image_tensor.append(self.image_transforms(img).unsqueeze(0).to(device))

        pixel_values = torch.cat(image_tensor)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float().half()
        latents =  torch.nn.functional.interpolate(pixel_values, size=(64, 64))
        return latents

    def encode_mask(self, character, device):
        
        _,locate = self.font_devise_64(character,True)
        if locate:
            chars_x,chars_y,chars_w,chars_h = locate
            a1 = chars_x
            a2 = chars_y
            a3 = chars_x+chars_w
            a4 = chars_y+chars_h
            mask_img = np.zeros((64, 64))
            polygon = np.array([[a1,a2],[a3,a2],[a3,a4],[a1,a4]],dtype = np.int32)
            mask_img = cv2.fillConvexPoly(mask_img, polygon , (1, 1, 1))
            mask_img = Image.fromarray(mask_img)
            mask_img_resize = transforms.Resize((64, 64), interpolation=transforms.InterpolationMode.NEAREST)(mask_img)
        else:
            mask_img_resize = np.zeros((64, 64))
        mask_tensor_resize = 1 - transforms.ToTensor()(mask_img_resize)
     
        return mask_tensor_resize.to(device)

    def encode_images_mask(self, character, device):
        if isinstance(character, str):
            return self.encode_mask(character, device)
        else:
            masks = []
            for prompt in character:
                mask_tensor_resize = self.encode_mask(prompt, device)
                masks.append(mask_tensor_resize.unsqueeze(1))
            return torch.cat(masks)

    def encode_prompt(self, prompts,font, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt):
        batch_size = len(prompts) if isinstance(prompts, list) else 1
        self.text_encoder = self.text_encoder.to(device)
        text_embeddings = self.text_encoder._encode_prompt(prompts,device=device,num_images_per_prompt=1,do_classifier_free_guidance=False)
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

            uncond_embeddings = self.text_encoder._encode_prompt(uncond_tokens,device=device,num_images_per_prompt=1,do_classifier_free_guidance=False)

            # duplicate unconditional embeddings for each generation per prompts, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_images_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_images_per_prompt, seq_len, -1)

            image_embedding = self.encode_images(font,device)
            co_embedding = torch.cat([text_embeddings,image_embedding.half()],1)
            co_embedding_convert = self.causal_transformer(co_embedding)
            uncond_embeddings = torch.nn.functional.pad(uncond_embeddings, pad=(0, 0, 0, 256), mode='constant', value=0) # batch*65*1024
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
        image_guidance_scale: float = 1.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        **kwargs,
    ):

        font = []
        prompt = []
        for t in prompts:
            font_sin = self.pattern.findall(t)
            if font_sin:
                font_sin = font_sin[0]
            else:
                font_sin = ""
            font.append(font_sin)
            prompt.append(t.replace(font_sin,""))


        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self.pipe._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        text_embeddings = self.encode_prompt(prompt,font, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt).half()
  
        self.pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.pipe.scheduler.timesteps

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

        extra_step_kwargs = self.pipe.prepare_extra_step_kwargs(generator, eta)

        latents_c1 = self.encode_images_vae(font,device)
        if self.random_mask:
            latents_c2 = self.encode_images_mask(font,device)
        else:
            latents_c2 = torch.zeros_like(latents_c1).to(device)
        font_latents = torch.cat([latents_c1,latents_c2],1)


        uncond_image_latents = torch.zeros_like(font_latents).to(device)
        font_latents = torch.cat([font_latents, font_latents, uncond_image_latents], dim=0)
        latents_copy = latents.clone()
        if not self.random_mask:
            for i, t in enumerate(self.pipe.progress_bar(timesteps)):
                latent_model_input = torch.cat([latents] * 3) if do_classifier_free_guidance else latents
                latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t)
                latent_model_input = torch.cat([latent_model_input, font_latents], dim=1)
                noise_pred = self.pipe.unet(latent_model_input.half(), t, encoder_hidden_states=text_embeddings).sample

                if do_classifier_free_guidance:
                    noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
                    noise_pred = (
                                noise_pred_uncond
                                + guidance_scale * (noise_pred_text - noise_pred_image)
                                + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
                            )
                latents = self.pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                if i==20:
                    features, encoded_edge_maps, noise_levels = [], [], []
                    latents_c1s = []
                    for ii in range(batch_size):
                        noisy_latent = latents[ii].unsqueeze(0) 
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
                        sqrt_one_minus_alpha_prod = (1 - self.pipe.scheduler.alphas_cumprod[t]).to(latents.device) ** 0.5
                        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
                        while len(sqrt_one_minus_alpha_prod.shape) < len(noisy_latent.shape):
                            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
                        noise_level = noise_pred[ii].unsqueeze(0)*sqrt_one_minus_alpha_prod
                        noise_level =  noise_level.transpose(1,3)
                        features.append(feature.unsqueeze(0))
                        noise_levels.append(noise_level.unsqueeze(0))
                        latents_c1s.append(latents_c1)
                    features = torch.cat(features)
                    noise_levels = torch.cat(noise_levels)
                    latents_c1s = torch.cat(latents_c1s)

                    pred_edge_map = self.model_mpm(features, noise_levels).unflatten(0, (batch_size, 64, 64)).transpose(3, 1)
                    pred_edge1 = torch.gt(pred_edge_map, 0.5)
                    pred_edge2 = torch.tensor(pred_edge1,dtype=torch.float)
                    pred_edge3 = pred_edge2[:,0].unsqueeze(1)
                    pred_edge = self.add_random(pred_edge3,font,device)
                    font_latents_lgb = torch.cat([latents_c1s,pred_edge],1)
                    uncond_image_latents = torch.zeros_like(font_latents_lgb).to(device)
                    font_latents = torch.cat([font_latents_lgb, font_latents_lgb, uncond_image_latents], dim=0)
                    break

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

        image = self.pipe.decode_latents(latents_copy)

        if output_type == "pil":
            image = self.pipe.numpy_to_pil(image)

        return image

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,  default="checkpoints/GlyphDraw_en", help="GlyphDraw model folder")
    parser.add_argument("--clip_path", type=str,  default="checkpoints/ViT-H-14.pt", help="clip model folder")
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--fonts", type=str, default="OPPOSans-S-B-0621.ttf", help="System font generation, which needs to be placed in the specified system file")
    parser.add_argument("--device", type=str,  default="cuda")
    parser.add_argument("--random_mask", type=bool,  default=False, help="Is the mask given randomly during inference")
    args = parser.parse_args()
    
    # ids = "39000"  # 39000*22*3*8/1000000=20 epoch
    # save_file = "font_results_acc/font_en_512_lmp"
    # files = "results_mul/stablediffusion_en"
    # model_id = "/public_data/ma/stable_models/model_font_6_en"  # model_font_6
    # font_unet_bin = f"/public_data/ma/code/stablediffusion-controlnet/{files}/unet_0_{ids}/pytorch_model.bin"
    # os.system("cp {} {}".format(font_unet_bin,os.path.join(model_id,"unet/diffusion_pytorch_model.bin")))
    # transformer_id = f"/public_data/ma/code/stablediffusion-controlnet/{files}/transformer_0_{ids}/pytorch_model.bin"
    # proj_id = f"/public_data/ma/code/stablediffusion-controlnet/{files}/proj_0_{ids}/pytorch_model.bin"
    # mpm_id = "/public_data/ma/code/stablediffusion-font/result/stablediffusion_mpm/mpm_0_9999/pytorch_model.bin"

    model_id = args.model_path
    transformer_id = os.path.join(model_id,"transformer_id.bin")
    proj_id = os.path.join(model_id,"projection_id.bin")
    mpm_id = os.path.join(model_id,"mpm_id.bin")
    sdt = StableDiffusionTest(args.clip_path, model_id,transformer_id,proj_id,mpm_id,args.device,args.random_mask,args.fonts)
    while True:
        raw_text = input("\nPlease Input Query (stop to exit) >>> ")
        if not raw_text:
            print('Query should not be empty!')
            continue
        if raw_text == "stop":
            break
        images = sdt([raw_text]*args.batch_size)
        for i, image in enumerate(images):
            image.save(f"{i}_new.png")
        grid = image_grid(images, rows=1, cols=args.batch_size)
        grid.save("glyphdraw.png")

import os
import torch
import argparse
from pytorch_lightning import (
    LightningModule,
    Trainer,
)
from cn_clip.clip import FullTokenizer
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
)
# from universal_datamodule import UniversalDataModule
from model_utils import (
    add_module_args,
    configure_optimizers,
    get_total_steps,
)
import transformers.optimization as optim
from universal_checkpoint import UniversalCheckpoint
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel, EulerDiscreteScheduler,PNDMScheduler
from torch.nn import functional as F
from tqdm.auto import tqdm
from universal_datamodule_lgp import DataModuleCustom
from typing import Callable, List, Optional, Union
from utils_new import load_config, load_clip, tokenize, save_config, save_model, numpy_to_pil, save_images
import inspect
from torch import nn
import math

# print(device)
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

    def init_weights(self, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

    def forward(self, x, t):
        # Concatenate input pixels with noise level t and positional encodings
        pos_encoding = [torch.sin(2 * math.pi * t * (2 **-l)) for l in range(self.num_encodings)]
        pos_encoding = torch.cat(pos_encoding, dim=-1)
        x = torch.cat((x, t, pos_encoding), dim=-1)
        x = x.flatten(start_dim=0, end_dim=3).half()
        
        return self.layers(x)

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
    # print("hooker working")
    save_tensors(self, out, 'activations')
    return out

class StableDiffusion(LightningModule):
    @staticmethod
    def add_module_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('OPPO Stable Diffusion Module')
        parser.add_argument('--train_whole_model', default=False)
        parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
        return parent_parser

    def __init__(self, args):
        super().__init__()
        self.tokenizer = FullTokenizer()
        self.bert_config = load_config(args.clip_path)
        self.text_encoder = load_clip(args.clip_path, self.bert_config)
        self.vae = AutoencoderKL.from_pretrained(args.model_path, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(args.model_path, subfolder="unet",torch_dtype=torch.float16)
        # self.test_scheduler = EulerDiscreteScheduler.from_pretrained(args.model_path, subfolder="scheduler",prediction_type="v_prediction")
        self.test_scheduler = PNDMScheduler.from_pretrained(args.model_path, subfolder="scheduler")
        self.test_scheduler.set_timesteps(50)
        # self.noise_scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
        self.save_hyperparameters(args)
        self.LGP = latent_guidance_predictor(output_dim=4, input_dim=7080, num_encodings=9).to("cuda")

        offset = self.test_scheduler.config.get("steps_offset", 0)
        accepts_eta = "eta" in set(inspect.signature(self.test_scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = 0.0

       

    def setup(self, stage) -> None:
        if stage == 'fit':
            self.total_steps = 10^9
            print('Total steps: {}' .format(self.total_steps))

    # def configure_optimizers(self):
    #     model_params = [{'params': self.LGP.parameters()}]
    #     return configure_optimizers(self, model_params=model_params)
    
    def configure_optimizers(self):
        weight_decay = 1e-6  #
        optimizer = torch.optim.Adam(self.LGP.parameters(), lr=0.0002, weight_decay=weight_decay)
        StepLR = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200], gamma=0.5)
        optim_dict = {'optimizer': optimizer, 'lr_scheduler': StepLR}
        return optim_dict

    @torch.no_grad()
    def encode_prompt(self, prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt):
        batch_size = len(prompt) if isinstance(prompt, list) else 1
        text_input_ids = tokenize(self.tokenizer, prompt)

        pad_index = self.tokenizer.vocab['[PAD]']
        attention_mask = text_input_ids.ne(pad_index).type(self.text_encoder.embeddings.word_embeddings.weight.dtype).to(device)

        text_embeddings = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        text_embeddings = text_embeddings[0]

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

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_images_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_images_per_prompt, seq_len, -1)

            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

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
        prompt: Union[str, List[str]],
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        num_images_per_prompt: Optional[int] = 1,
        **kwargs,
    ):
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        do_classifier_free_guidance = guidance_scale > 1.0

        text_embeddings = self.encode_prompt(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        ).half()

        self.test_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.test_scheduler.timesteps

        shape = (batch_size * num_images_per_prompt, self.unet.in_channels, height // 8, width // 8)
        latents = torch.randn(shape, generator=generator, device=device, dtype=text_embeddings.dtype).half()
        latents = latents * self.test_scheduler.init_noise_sigma

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        for i, t in enumerate(tqdm(timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.test_scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.test_scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = numpy_to_pil(image)

        return image

    def training_step(self, batch, batch_idx):
        self.LGP.train()
        bsz = len(batch["texts"])

        with torch.no_grad():
            latents = self.vae.encode(batch["pixel_values"]).latent_dist.sample()
            latents = latents * 0.18215
        timesteps = torch.randint(400, 700, (1,), device=latents.device)
        timesteps = timesteps.long()
        # Sample noise that we'll add to the latents
        noise = torch.randn(latents.shape).to(latents.device)
        noise = noise.to(dtype=self.unet.dtype)

        noisy_latents = self.test_scheduler.add_noise(latents, noise, timesteps)
        noisy_latents = noisy_latents.to(dtype=self.unet.dtype)

        features, encoded_edge_maps, noise_levels = [], [], []
        for i in range(bsz):
            # Predict the noise residual
            text_embeddings = self.encode_prompt(batch["texts"][i], latents.device, 1, True, None).half()
            noisy_latent = noisy_latents[i].unsqueeze(0)
            latent = latents[i].unsqueeze(0)
            latent_model_input = torch.cat([noisy_latent] * 2)
            
            save_hook = save_out_hook
            blocks = [0,1,2,3]
            self.feature_blocks = []
            for idx, block in enumerate(self.unet.down_blocks):
                if idx in blocks:
                    h=block.register_forward_hook(save_hook)
                    self.feature_blocks.append([block,h]) 
                    
            for idx, block in enumerate(self.unet.up_blocks):
                if idx in blocks:
                    h=block.register_forward_hook(save_hook)
                    self.feature_blocks.append([block,h])  

            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, timesteps, encoder_hidden_states=text_embeddings).sample

            # Extract activations
            activations = []
            for block,h in self.feature_blocks:
                activations.append(block.activations)
                block.activations = None
                h.remove()
                
            activations = [activations[0][0], activations[1][0], activations[2][0], activations[3][0], activations[4], activations[5], activations[6], activations[7]]
        
            # lgb predict
            feature =  resize_and_concatenate(activations, latent)
            sqrt_alpha_prod = self.test_scheduler.alphas_cumprod[timesteps].to(latents.device) ** 0.5
            sqrt_alpha_prod = sqrt_alpha_prod.flatten()
            while len(sqrt_alpha_prod.shape) < len(latent.shape):
                sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            noise_level = noisy_latent - (sqrt_alpha_prod * latent)### 任意时刻的x_t可以由x_0和\beta表示，noise_level就是x_t - sqrt_alpha_prod * x_0
            noise_level =  noise_level.transpose(1,3)
            encoded_edge_map = batch["mask"][i].unsqueeze(0).transpose(1,3)
            encoded_edge_map = torch.nn.functional.pad(encoded_edge_map, pad=(3,0,0,0), mode='replicate')
            encoded_edge_map = encoded_edge_map.flatten(start_dim=0, end_dim=2)
            features.append(feature.unsqueeze(0))
            encoded_edge_maps.append(encoded_edge_map.unsqueeze(0))
            noise_levels.append(noise_level.unsqueeze(0))

        features = torch.cat(features)
        encoded_edge_maps = torch.cat(encoded_edge_maps)
        encoded_edge_maps = encoded_edge_maps.flatten(start_dim=0, end_dim=1)
        noise_levels = torch.cat(noise_levels)

        outputs = self.LGP(features,noise_levels)

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        loss = F.mse_loss(outputs, encoded_edge_maps, reduction="none").mean()
        self.log("train_loss", loss.item(),  on_epoch=False, prog_bar=True, logger=True)
        self.log("lr", lr,  on_epoch=False, prog_bar=True, logger=True)


        if self.trainer.global_rank == 0:
            if (self.global_step+1) % 2000 == 0:
                print('saving model...')
                save_path_lgp = os.path.join(args.default_root_dir, f'lgp_{self.trainer.current_epoch}_{self.global_step}')
                save_model(self.LGP, save_path_lgp)
                print(save_path_lgp)

        return {"loss": loss}
    
    # def save_model():

    def on_train_epoch_end(self):
        pass

    def on_load_checkpoint(self, checkpoint) -> None:
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
    tokenizer = model.tokenizer
    
    def collate_fn(examples):
        # print(examples)
        texts = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]
        font = [example["font"] for example in examples]
        mask = torch.stack([example["mask"] for example in examples])
        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        # input_ids = tokenizer.pad({"input_ids": input_ids}, padding=True,
        #                           return_tensors="pt").input_ids
        input_ids = tokenize(tokenizer, texts)
        batch = {
            "font": font,
            "texts": texts,
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "mask":mask
        }

        return batch

    datamoule = DataModuleCustom(
        args, tokenizer=tokenizer, collate_fn=collate_fn)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = UniversalCheckpoint(args)

    trainer = Trainer.from_argparse_args(args,
                                         callbacks=[
                                             lr_monitor,
                                             checkpoint_callback])

    trainer.fit(model, datamoule, ckpt_path=args.load_ckpt_path)

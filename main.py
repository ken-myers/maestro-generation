import argparse
import torch
from abc import ABC, abstractmethod
import json
from tqdm import tqdm
import os
import soundfile as sf
import math
import numpy as np
import warnings

warnings.filterwarnings(
    "ignore",
    message="Should have tb<=t1 but got tb=.*",
    module="torchsde._brownian.brownian_interval"
)

warnings.filterwarnings(
    "ignore",
    message="To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_.*",
    module="c.modeling_encodec"
)

warnings.filterwarnings(
    "ignore",
    message="To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).",
    module="transformers.models.encodec.modeling_encodec"
)

class AudioGenerator(ABC):

    @abstractmethod
    def generate(self, *, prompts, gens_per_prompt, duration, **kwargs):
        pass

class StableAudioGenerator(AudioGenerator):
    def __init__(self, device):
        from diffusers import StableAudioPipeline
        pipe = StableAudioPipeline.from_pretrained('stabilityai/stable-audio-open-1.0', torch_dtype=torch.float16)
        self.pipe = pipe.to(device)
        self.sr = self.pipe.vae.sampling_rate

    def generate(self, *, prompts, gens_per_prompt, duration, **kwargs):
        
        steps = kwargs.get('steps', 200)
        
        audios = self.pipe(
            prompts,
            negative_prompt = ["low quality, choppy, noisy audio"] * len(prompts),
            num_waveforms_per_prompt = gens_per_prompt,
            audio_end_in_s=duration,
            num_inference_steps=steps
        ).audios.permute(0, 2, 1)

        return audios

class MusicGenAudioGenerator(AudioGenerator):
    def __init__(self, device):
        from transformers import MusicgenForConditionalGeneration, AutoProcessor
        self.processor = AutoProcessor.from_pretrained('facebook/musicgen-large')
        self.model = MusicgenForConditionalGeneration.from_pretrained('facebook/musicgen-large', torch_dtype=torch.float16, attn_implementation="eager")
        self.model = self.model.to(device)
        self.sr = 32000
        self.tokens_per_second = 51.2
    
    def generate(self, *, prompts, gens_per_prompt, duration, **kwargs):

        max_tokens = math.ceil(duration * self.tokens_per_second)

        duped_prompts = [p for p in prompts for _ in range(gens_per_prompt)]

        inputs = self.processor(text=duped_prompts, return_tensors='pt', padding=True).to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True
        )
        
        return outputs.squeeze(1)
    

def load_generator(model_name, device):
    if model_name == 'stable':
        return StableAudioGenerator(device=device)
    elif model_name == 'musicgen':
        return MusicGenAudioGenerator(device=device)
    else:
        raise ValueError('Invalid model name. Choose either musicgen or stable.')

def generate(*, model_name, input_path, output_dir, batch_size, device, gens_per_prompt, file_prefix, **kwargs):
    
    #load model
    generator = load_generator(model_name, device)

    #load input json
    with open(input_path, 'r') as f:
        base_prompts = json.load(f)

    #flatten prompts
    prompts = []
    for prompt in base_prompts:
        genre = prompt['genre']
        variations = prompt['variations']

        for variation in variations:
            v = {
                'label': variation,
                'genre': genre
            }
            prompts.append(v)

    #chunk prompts into batches
    batches = [prompts[i:min(len(prompts), i+batch_size)] for i in range(0, len(prompts), batch_size)]
    

    for b_num, batch in enumerate(tqdm(batches, desc='Generating audio')):
        new_json = []

        text_prompts = [p['label'] for p in batch]
        audios = generator.generate(prompts=text_prompts, gens_per_prompt=gens_per_prompt, **kwargs).cpu().numpy()
        
        # print(f"audios shape: {audios.shape}, dtype: {audios.dtype}")

        audios = audios.astype(np.float32)

        for i, audio in enumerate(audios):
            #save audio
            filename = f'{file_prefix}B{b_num}_{i}.wav'
            filepath = os.path.join(output_dir, filename)
            sf.write(filepath, audio, generator.sr)

            #update json
            prompt_index = i // gens_per_prompt
            prompt = batch[prompt_index].copy()
            prompt['filename'] = filename
            prompt['source'] = model_name
            new_json.append(prompt)
        
        #save new json
        new_json_path = os.path.join(output_dir, f'{file_prefix}metadata-b{b_num}.json')
        with open(new_json_path, 'w') as f:
            json.dump(new_json, f)
        

            
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Input json path')
    parser.add_argument('--output', type=str, help='directory to output wavs')
    parser.add_argument('--model', type=str, help='model name, either musicgen or stable')
    parser.add_argument('--batch_size', type=int, default=10, help='batch size')
    parser.add_argument('--device', type=int, default=0, help='device to use')
    parser.add_argument('--gens_per_prompt', type=int, default=1, help='number of generations per prompt')
    parser.add_argument('--file_prefix', type=str, default=None, help='prefix for output files')
    parser.add_argument('--duration', type=float, default=30, help='duration of generated audio in seconds')
    
    
    args = parser.parse_args()

    if args.file_prefix is None:
        file_prefix = f"{args.device}-" 
    else:
        file_prefix = args.file_prefix

    generate(
        model_name=args.model,
        input_path=args.input,
        output_dir=args.output,
        batch_size=args.batch_size,
        device=args.device,
        gens_per_prompt=args.gens_per_prompt,
        file_prefix=file_prefix,
        duration=args.duration
    )

if __name__ == "__main__":
    main()
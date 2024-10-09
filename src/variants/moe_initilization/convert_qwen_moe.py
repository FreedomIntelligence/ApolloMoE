from modeling_upcycling_qwen2_moe import UpcyclingQwen2MoeForCausalLM, UpcyclingQwen2MoeConfig
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM,AutoModel
from transformers.generation import GenerationConfig
from transformers.trainer_utils import set_seed
import argparse, torch, shutil, os
import json
import torch.nn as nn
# torch.set_default_device("cuda")

seed = 1024
context_length_per_experiment = 1
generate_length_per_experiment = 256

set_seed(seed)

def convert(args):
    model_path=args.model_path
    output_path=args.output_path
    # Convert vanilla Qwen to UpcyclingQwen,
    model = UpcyclingQwen2MoeForCausalLM.from_qwen(model_path, torch_dtype="auto", trust_remote_code=True,share_flag=args.share_flag,attn_init_change=args.attn_init_change,language_gate=args.language_gate)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)


    print_trainable_parameters(model)
    print(model.config)

    print(model.state_dict().keys())

    print(f"saved at {output_path}")

def convert_btx(args):
    model_path=args.model_path
    output_path=args.output_path
    # Convert vanilla QWen to UpcyclingQWen, 

    model = UpcyclingQwen2MoeForCausalLM.from_qwen_btx(model_path, torch_dtype="auto", trust_remote_code=True,share_flag=args.share_flag)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    print_trainable_parameters(model)
    print(model.config)
    print(f"saved at {output_path}")

def convert_prune_btx(args):
    model_path=args.model_path
    output_path=args.output_path
    # Convert vanilla QWen to UpcyclingQWen, 
    layer_ids,expert_ids=prune_count(args.count_path,args.count_thresh)

    model = UpcyclingQwen2MoeForCausalLM.from_prune_btx(model_path, torch_dtype="auto", trust_remote_code=True,share_flag=args.share_flag,prune=args.prune,layer_ids=layer_ids,expert_ids=expert_ids)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    print_trainable_parameters(model)
    print(model.config)

    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)


    print(f"saved at {output_path}")

def load_model(args):    
    model_path = args.output_path
    print(f"model_path: {model_path}")
    # config = UpcyclingQwen2MoeConfig.from_pretrained(model_path, device_map='auto')

    # model = UpcyclingQwen2MoeForCausalLM.from_pretrained(model_path,trust_remote_code=True)
    # model=AutoModel.from_pretrained(model_path,trust_remote_code=True)
    model=AutoModelForCausalLM.from_pretrained(model_path,trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print(model.config)
    model.eval()

    generation_config,unused_kwargs = GenerationConfig.from_pretrained(args.model_path,pad_token_id=tokenizer.eos_token_id,num_return_sequences=1, max_new_tokens=256, min_new_tokens=2, do_sample=False, temperature=1.0, top_k=50, top_p=1.0,return_unused_kwargs=True)

    # generation_config,unused_kwargs = GenerationConfig.from_pretrained(model_path,pad_token_id=tokenizer.eos_token_id,do_sample=True,return_unused_kwargs=True)
    print('unuse_kawargs:',unused_kwargs)
    # generation_config.max_new_tokens = generate_length_per_experiment    
    model.generation_config = generation_config
    return model, tokenizer

def test(args):
    model, tokenizer = load_model(args)
    model_dict=model.state_dict()

    print('lm_head.weight',model_dict['lm_head.weight']) if 'lm_head.weight' in model_dict.keys() else print(f'No argument lm_head.weight')
            
    prompt= "Hi what's your name? And tell me a story about your name"
    
    model_inputs = tokenizer([prompt], return_tensors="pt")

    print(type(model_inputs))
    print(f'model_inputs:{model_inputs}')

    generated_ids = model.generate(model_inputs.input_ids)

    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(response)

def test1(args):
    model, tokenizer = load_model(args)
    print(model.config)

    prompt= "Hi what's your name? And tell me a story about your name"
    input_ids=tokenizer.encode(prompt,return_tensors='pt')

    generated_ids=model.generate(input_ids,pad_token_id=tokenizer.eos_token_id)
    text=tokenizer.decode(generated_ids[0],skip_special_token=True)
    print(prompt)
    print(text[len(prompt):])



def test2(args):
    model, tokenizer = load_model(args)

    def reinit_weights(m):
        if isinstance(m, nn.Linear): 
            nn.init.constant_(m.weight, 0.0)
            # nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    print(type(model.named_modules),model.named_modules())

    for name,param in model.named_parameters():
        if '.mlp.gate' in name:
            print(param)
            break

    for name,module in model.named_modules():
        if '.mlp.gate' in name:
            module.apply(reinit_weights)

    for name,param in model.named_parameters():
        if '.mlp.gate' in name:
            print(param)
            break

def test3(args):
    model, tokenizer = load_model(args)
    def reinit_weights(m):
        if isinstance(m, nn.Linear): 
            nn.init.constant_(m.weight, 0.0)
            # nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    print(type(model.named_modules),model.named_modules())

    for name,param in model.named_parameters():
        if '.mlp.gate' in name:
            print(param)
            break

    for name,module in model.named_modules():
        if '.mlp.gate' in name:
            if isinstance(module,nn.Linear):
                module.weight.data.zero_()
                # module.weight.data.normal_(mean=0,std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()

    for name,param in model.named_parameters():
        if '.mlp.gate' in name:
            print(param)
            break

def print_trainable_parameters(model, desc:str=""):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes
        # one needs to multiply the number of parameters by 2 to get
        # the correct number of parameters
        if param.__class__.__name__ == "Params4bit":
            num_params = num_params * 2

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    
    desc = f"[{desc}] " if len(desc) > 0 else ""
    print(
        f"{desc}trainable params: {trainable_params}({trainable_params/1e9:.1f}B) || all params: {all_param}({all_param/1e9:.1f}B) || trainable%: {100 * trainable_params / all_param}"
    )
    return trainable_params, all_param


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', '-t', action='store_true')
    parser.add_argument('--model_path', '-m', type=str)
    parser.add_argument('--output_path', '-o', type=str)
    parser.add_argument('--share_flag','-s',action='store_true')
    parser.add_argument('--attn_init_change','-a',action='store_true')
    parser.add_argument('--language_gate',action='store_true')

    
    args = parser.parse_args()

    if args.test:
        test1(args)
    else:
        convert(args)
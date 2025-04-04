import argparse
import os
import torch
from PIL import Image
import requests
from io import BytesIO
import gc
from transformers import AutoTokenizer
from llava.model.language_model.llava_phi3 import LlavaPhiForCausalLM, LlavaPhiConfig
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava import conversation as conversation_lib
from llava.model.builder import load_pretrained_model
from llava.peft import LoraConfig, get_peft_model

def load_image(image_path):
    """Load an image from a file path or URL"""
    if image_path.startswith(('http://', 'https://')):
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(image_path)
    return image.convert('RGB')

def find_all_linear_names(model):
    """Find all linear layer names in the model"""
    linear_names = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            linear_names.append(name)
    return linear_names

def main():
    parser = argparse.ArgumentParser(description="HealthGPT Comprehension Inference")
    
    # Model configuration
    parser.add_argument("--model_name_or_path", type=str, default="microsoft/Phi-3-mini-4k-instruct",
                        help="Path to the base model or model identifier from huggingface.co/models")
    parser.add_argument("--dtype", type=str, default="FP16", choices=["FP16", "BF16", "FP32"],
                        help="Data type for model inference")
    parser.add_argument("--hlora_r", type=int, default=64,
                        help="H-LoRA rank parameter")
    parser.add_argument("--hlora_alpha", type=int, default=128,
                        help="H-LoRA alpha parameter")
    parser.add_argument("--hlora_nums", type=int, default=4,
                        help="Number of H-LoRA layers")
    parser.add_argument("--vq_idx_nums", type=int, default=8192,
                        help="Number of VQ indices")
    parser.add_argument("--instruct_template", type=str, default="phi3_instruct",
                        help="Instruction template for the model")
    
    # Memory management
    parser.add_argument("--max_memory", type=dict, default=None,
                        help="Max memory per GPU device")
    parser.add_argument("--offload_folder", type=str, default="offload",
                        help="Directory to offload weights")
    
    # Paths
    parser.add_argument("--vit_path", type=str, required=True,
                        help="Path to the Vision Transformer model")
    parser.add_argument("--hlora_path", type=str, required=True,
                        help="Path to the H-LoRA weights for comprehension")
    parser.add_argument("--fusion_layer_path", type=str,
                        help="Path to the fusion layer weights")
    
    # Input
    parser.add_argument("--question", type=str, required=True,
                        help="Question to ask about the image")
    parser.add_argument("--img_path", type=str, required=True,
                        help="Path to the input image")
    
    args = parser.parse_args()
    
    # Create offload directory if needed
    if not os.path.exists(args.offload_folder):
        os.makedirs(args.offload_folder)
    
    # Set device and memory management
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Set default max memory if not provided
    if args.max_memory is None and device == "cuda":
        total_mem = torch.cuda.get_device_properties(0).total_memory
        args.max_memory = {0: f"{int(total_mem * 0.8 / 1024**3)}GiB"}  # Use 80% of GPU memory
    
    # Clear CUDA cache
    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()
    
    # Set data type
    if args.dtype == "FP16":
        dtype = torch.float16
    elif args.dtype == "BF16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    
    try:
        # Load tokenizer first
        print(f"Loading tokenizer from {args.model_name_or_path}...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        
        # Create LLaVA config
        print("Creating LLaVA configuration...")
        config = LlavaPhiConfig.from_pretrained(args.model_name_or_path)
        config.mm_vision_tower = args.vit_path
        config.mm_hidden_size = 1024  # ViT-L/14 hidden size
        config.mm_vision_select_layer = -2
        config.mm_vision_select_feature = "patch"
        
        # Load base model with LLaVA config and memory optimization
        print(f"Loading model from {args.model_name_or_path}...")
        model = LlavaPhiForCausalLM(config)
        
        # Load model with memory optimization
        if device == "cuda":
            model = model.to(dtype)  # Convert to lower precision first
            if args.max_memory:
                # Load model with memory optimization
                model = model.to_bettertransformer()  # Use more memory efficient attention
                model.enable_input_require_grads()  # Enable gradient checkpointing
        
        # Load base model weights
        print("Loading base model weights...")
        model.load_state_dict(
            torch.load(
                args.model_name_or_path + "/pytorch_model.bin",
                map_location='cpu'
            ),
            strict=False
        )
        
        # Configure and apply H-LoRA with memory optimization
        print(f"Configuring H-LoRA with rank={args.hlora_r}, alpha={args.hlora_alpha}...")
        lora_config = LoraConfig(
            r=args.hlora_r,
            lora_alpha=args.hlora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=0.0,
            bias='none',
            task_type="CAUSAL_LM",
            lora_nums=args.hlora_nums,
            inference_mode=True,  # Enable memory optimization for inference
        )
        model = get_peft_model(model, lora_config)
        
        # Initialize vision modules
        print("Initializing vision modules...")
        model.get_model().initialize_vision_modules(model_args=args)
        
        # Load H-LoRA weights
        print(f"Loading H-LoRA weights from {args.hlora_path}...")
        hlora_state_dict = torch.load(args.hlora_path, map_location='cpu')
        model.load_state_dict(hlora_state_dict, strict=False)
        
        # Load fusion layer if provided
        if args.fusion_layer_path:
            print(f"Loading fusion layer weights from {args.fusion_layer_path}...")
            fusion_state_dict = torch.load(args.fusion_layer_path, map_location='cpu')
            model.get_model().mm_projector.load_state_dict(fusion_state_dict)
        
        model.eval()
        
        # Load and process image
        print(f"Loading image from {args.img_path}...")
        image = load_image(args.img_path)
        
        # Clear some memory before processing
        if device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
        
        # Prepare input with image token
        print("Preparing input...")
        question_text = DEFAULT_IMAGE_TOKEN + '\n' + args.question
        conv = conversation_lib.conv_templates[args.instruct_template].copy()
        conv.append_message(conv.roles[0], question_text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        # Tokenize input
        input_ids = tokenizer_image_token(
            prompt, 
            tokenizer, 
            IMAGE_TOKEN_INDEX, 
            return_tensors='pt'
        ).to(device)
        
        # Process image
        vision_tower = model.get_model().get_vision_tower()
        image_tensor = vision_tower.image_processor.preprocess(
            image, 
            return_tensors='pt'
        )['pixel_values'][0]
        
        # Generate response with memory optimization
        print("Generating response...")
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids.unsqueeze(0),
                images=image_tensor.unsqueeze(0).to(dtype=dtype, device=device),
                do_sample=False,
                temperature=0.0,
                max_new_tokens=512,
                use_cache=True,
                max_memory=args.max_memory,
            )
        
        response = tokenizer.decode(
            output_ids[0], 
            skip_special_tokens=True
        )[len(prompt):]
        
        print("\nResponse:")
        print(response.strip())
        
    finally:
        # Clean up
        if device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
        
        # Delete model and clear memory
        if 'model' in locals():
            del model
        if 'vision_tower' in locals():
            del vision_tower
        if 'image_tensor' in locals():
            del image_tensor
        if 'output_ids' in locals():
            del output_ids

if __name__ == "__main__":
    main() 
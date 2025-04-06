import os
import sys
import torch
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import logging
from PIL import Image
import io
from typing import Optional
from transformers import AutoTokenizer, CLIPImageProcessor
from llava.model.language_model.llava_phi3 import LlavaPhiForCausalLM, LlavaPhiConfig
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.mm_utils import tokenizer_image_token
from llava import conversation as conversation_lib
from llava.demo.utils import find_all_linear_names, add_special_tokens_and_resize_model, load_weights, expand2square, com_vision_args, gen_vision_args
from llava.peft import LoraConfig, get_peft_model

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#global variables for model and tokenizer
model = None
tokenizer = None
image_processor = None
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

def load_model():
    global model, tokenizer, image_processor
    
    try:
        logger.info("Starting model loading...")
        
        model_name = "microsoft/Phi-3-mini-4k-instruct"
        vision_tower = "openai/clip-vit-large-patch14-336"
        hlora_path = "models/HealthGPT-M3/com_hlora_weights.bin"
        fusion_layer_path = "models/HealthGPT-M3/fusion_layer_weights.bin"
        
        #load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="right",
            use_fast=False,
        )
        
        # load model
        model = LlavaPhiForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype
        )
        
        # configure H-LoRA
        lora_config = LoraConfig(
            r=64,
            lora_alpha=128,
            target_modules=find_all_linear_names(model),
            lora_dropout=0.0,
            bias='none',
            task_type="CAUSAL_LM",
            lora_nums=4,
        )
        model = get_peft_model(model, lora_config)
        
        # add special tokens first
        logger.info(f"Initial vocabulary size: {len(tokenizer)}")
        special_tokens = {
            "pad_token": "[PAD]",
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>"
        }
        tokenizer.add_special_tokens(special_tokens)
        logger.info(f"Vocabulary size after special tokens: {len(tokenizer)}")
        
        # now add exactly enough tokens to reach target size
        target_size = 40206
        current_size = len(tokenizer)
        tokens_to_add = [f"[V{i}]" for i in range(target_size - current_size)]
        tokenizer.add_tokens(tokens_to_add)
        logger.info(f"Final vocabulary size: {len(tokenizer)}")
        
        # resize model embeddings to match tokenizer
        model.resize_token_embeddings(len(tokenizer))
        logger.info(f"Model embeddings resized to: {model.get_input_embeddings().weight.shape}")
        
        # now load H-LoRA and fusion weights
        logger.info("Loading H-LoRA and fusion layer weights...")
        model = load_weights(model, hlora_path, fusion_layer_path)
        
        # initialize vision modules with generation config by default
        logger.info("initializing vision modules...")
        from llava.demo.utils import gen_vision_args
        
        # set up vision args for generation
        gen_vision_args.model_name_or_path = model_name
        gen_vision_args.vision_tower = "openai/clip-vit-large-patch14-336"
        gen_vision_args.version = "phi3_instruct"
        gen_vision_args.mm_vision_select_layer = -2  # Important for generation
        gen_vision_args.mm_vision_select_feature = "patch"  # Important for generation
        
        logger.info(f"Vision args: {vars(gen_vision_args)}")
        model.get_model().initialize_vision_modules(model_args=gen_vision_args)
        logger.info("Vision modules initialized")
        
        # get vision tower after initialization
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=dtype, device=device)
        
        # move model to device and set eval mode
        model = model.to(dtype=dtype, device=device)
        model.eval()
        
        # load image processor using the vision tower path
        image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        
        logger.info("Model loading completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        return False

@app.on_event("startup")
async def startup_event():
    load_model()

@app.post("/api/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    task_type: str = Form(...),
    question: Optional[str] = Form(None)
):
    try:
        logger.info(f"Starting {task_type} task...")
        
        # Read and process image
        logger.info("Reading image file...")
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Preprocess image
        logger.info("Preprocessing image...")
        image = expand2square(image, tuple(int(x*255) for x in model.get_vision_tower().image_processor.image_mean))
        image_tensor = model.get_vision_tower().image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        image_tensor = image_tensor.to(device=device, dtype=dtype)
        logger.info(f"Image preprocessed, tensor shape: {image_tensor.shape}")
        
        # Set up generation parameters once
        if task_type == "generation":
            logger.info("Setting up generation parameters...")
            generation_kwargs = {
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
                "num_beams": 1,
                "max_new_tokens": 1100,
                "use_cache": True,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id
            }
            # Use the exact prompt format from gen_infer.py
            if not question:
                question = "Generate a high-quality image based on this one."

            question = question + " <start_index>"
        else:
            logger.info("Setting up comprehension parameters...")
            generation_kwargs = {
                "do_sample": False,
                "num_beams": 1,
                "max_new_tokens": 512,
                "use_cache": True,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id
            }
            if not question:
                question = "Please analyze this image."
        
        logger.info("Preparing conversation...")
        qs = DEFAULT_IMAGE_TOKEN + '\n' + question
        conv = conversation_lib.conv_templates["phi3_instruct"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        logger.info(f"Prompt prepared: {prompt[:100]}...")
        
        # Tokenize input
        logger.info("Tokenizing input...")
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        input_ids = input_ids.unsqueeze_(0).to(device=device, dtype=torch.long)
        logger.info(f"Input shapes - input_ids: {input_ids.shape}, image_tensor: {image_tensor.shape}")
        
        # Generate response
        logger.info(f"Generating with parameters: {generation_kwargs}")
        with torch.inference_mode():
            try:
                output_ids = model.base_model.model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0),
                    image_sizes=image.size,
                    **generation_kwargs
                )
                logger.info("Generation completed successfully")

                decoded_start = tokenizer.decode(output_ids[0][:50], skip_special_tokens=True)
                logger.info(f"Start of generated text: {decoded_start}")
            except Exception as e:
                logger.error(f"Error during generation: {str(e)}")
                logger.error(f"Input shapes - input_ids: {input_ids.shape}, image_tensor: {image_tensor.shape}")
                logger.error(f"Input dtypes - input_ids: {input_ids.dtype}, image_tensor: {image_tensor.dtype}")
                raise e
        
        # Process response
        logger.info("Processing response...")
        if task_type == "comprehension":

            full_response = tokenizer.decode(output_ids[0], skip_special_tokens=True)[:-8]
            
            # Ensure response ends on a complete sentence
            sentence_endings = ['. ', '! ', '? ']
            last_end = -1
            for ending in sentence_endings:
                pos = full_response.rfind(ending)
                last_end = max(last_end, pos)
            
            if last_end != -1:
                response = full_response[:last_end + 1].strip()
            else:
                response = full_response.strip()
                
            logger.info("=== FULL RESPONSE ===")
            logger.info(response)
            logger.info("=== END RESPONSE ===")

            return {
                "success": True, 
                "response": response,
                "full_response": response
            }
        else:
            try:
                logger.info("Processing generated image...")
                import re
                from taming_transformers.idx2img import idx2img
                
                decoded_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)[:-8]
                logger.info(f"Decoded text: {decoded_text[:100]}...")
                response = [int(idx) for idx in re.findall(r'\d+', decoded_text)]
                logger.info(f"Extracted {len(response)} indices from response")
                logger.info(f"First 10 indices: {response[:10]}")
                
                # convert indices to image
                save_path = "temp/generated.jpg"
                os.makedirs("temp", exist_ok=True)
                
                if len(response) < 1024:  # minimum required for idx2img
                    raise ValueError(f"Not enough indices for image generation. Got {len(response)}, need at least 1024")
                
                # convert to tensor and generate image
                indices_tensor = torch.tensor(response).cuda()
                logger.info(f"converting indices to image with shape {indices_tensor.shape}")
                idx2img(indices_tensor, save_path)
                
                logger.info(f"Image saved to {save_path}")
                return {"success": True, "image_path": save_path}
            except Exception as e:
                logger.error(f"Error processing generated image: {str(e)}")
                logger.error(f"Response length: {len(response) if 'response' in locals() else 'N/A'}")
                return {"success": False, "error": f"Failed to process generated image: {str(e)}"}
            
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        return {"success": False, "error": str(e)}

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

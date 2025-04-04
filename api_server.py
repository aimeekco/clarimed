import os
import subprocess
import tempfile
import logging
import psutil
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
from PIL import Image
import io
import shutil
import time
import threading
import gc

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="HealthGPT Medical Imaging API")

# Add CORS middleware with more permissive settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Create directories for temporary files and offload
os.makedirs("temp", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("offload", exist_ok=True)

# Global variables to track model loading status
model_loading = False
model_loaded = False
model_load_start_time = 0

def get_available_memory():
    """Get available system memory in GB"""
    vm = psutil.virtual_memory()
    return vm.available / (1024 ** 3)  # Convert to GB

def get_gpu_memory():
    """Get available GPU memory in GB"""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        return None
    except:
        return None

def cleanup_memory():
    """Clean up system and GPU memory"""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass

def load_model_async():
    """Load the model asynchronously to avoid blocking the server startup"""
    global model_loading, model_loaded, model_load_start_time
    
    model_loading = True
    model_load_start_time = time.time()
    logger.info("Starting asynchronous model loading...")
    
    try:
        # Clean up before loading
        cleanup_memory()
        
        # Calculate memory limits
        available_memory = get_available_memory()
        gpu_memory = get_gpu_memory()
        
        # Set memory limits (80% of available memory)
        memory_limit = int(min(available_memory, gpu_memory or float('inf')) * 0.8 * 1024) if gpu_memory else None
        
        # Create a test image if it doesn't exist
        if not os.path.exists("temp/test.jpg"):
            img = Image.new('RGB', (100, 100), color='white')
            img.save("temp/test.jpg")
        
        # Base command with memory optimizations
        cmd = [
            "python", "com_infer.py",
            "--model_name_or_path", "microsoft/Phi-3-mini-4k-instruct",
            "--dtype", "FP16",
            "--hlora_r", "64",
            "--hlora_alpha", "128",
            "--hlora_nums", "4",
            "--vq_idx_nums", "8192",
            "--instruct_template", "phi3_instruct",
            "--vit_path", "models/ViT",
            "--hlora_path", f"models/HealthGPT-M3/com_hlora_weights.bin",
            "--fusion_layer_path", f"models/HealthGPT-M3/fusion_layer_weights.bin",
            "--offload_folder", "offload",
            "--question", "Test question",
            "--img_path", "temp/test.jpg"
        ]
        
        # Add memory limit if available
        if memory_limit:
            cmd.extend(["--max_memory", str(memory_limit)])
        
        logger.info(f"Running with memory limit: {memory_limit}MB" if memory_limit else "Running without memory limit")
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Set a longer timeout for the first load (60 minutes)
        stdout, stderr = process.communicate(timeout=3600)  # 60-minute timeout
        
        if process.returncode != 0:
            logger.error(f"Model loading failed: {stderr}")
        else:
            logger.info("Model loaded successfully")
            model_loaded = True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
    finally:
        cleanup_memory()
        model_loading = False
        logger.info(f"Model loading completed in {time.time() - model_load_start_time:.2f} seconds")

# Start model loading in a separate thread
threading.Thread(target=load_model_async, daemon=True).start()

@app.get("/api/model-status")
async def model_status():
    """Check the status of model loading"""
    if model_loaded:
        return {"status": "loaded", "message": "Model is loaded and ready"}
    elif model_loading:
        elapsed = time.time() - model_load_start_time
        return {
            "status": "loading", 
            "message": f"Model is loading (started {elapsed:.0f} seconds ago)",
            "elapsed_seconds": elapsed
        }
    else:
        return {"status": "error", "message": "Model loading failed or not started"}

@app.post("/api/process-scan")
async def process_scan(
    scan: UploadFile = File(...),
    taskType: str = Form(...)
):
    """
    Process a medical scan based on the specified task type.
    
    - taskType: "enhance", "ct2mri", "mri2ct", or "diagnosis"
    """
    global model_loading, model_loaded
    
    # Check if model is still loading
    if model_loading:
        elapsed = time.time() - model_load_start_time
        if elapsed > 3600:  # If loading for more than 60 minutes
            return JSONResponse(
                status_code=503,
                content={
                    "detail": "Model is still loading. Please try again later.",
                    "elapsed_seconds": elapsed
                }
            )
        else:
            return JSONResponse(
                status_code=503,
                content={
                    "detail": "Model is still loading. Please wait.",
                    "elapsed_seconds": elapsed
                }
            )
    
    # If model failed to load
    if not model_loaded:
        return JSONResponse(
            status_code=503,
            content={"detail": "Model failed to load. Please restart the server."}
        )
    
    start_time = time.time()
    temp_file_path = None
    
    try:
        # Clean up before processing
        cleanup_memory()
        
        # Calculate memory limits
        available_memory = get_available_memory()
        gpu_memory = get_gpu_memory()
        memory_limit = int(min(available_memory, gpu_memory or float('inf')) * 0.8 * 1024) if gpu_memory else None
        
        # Validate task type
        valid_tasks = ["enhance", "ct2mri", "mri2ct", "diagnosis"]
        if taskType not in valid_tasks:
            raise HTTPException(status_code=400, detail=f"Invalid task type. Must be one of: {', '.join(valid_tasks)}")
        
        # Save uploaded file to temporary location
        temp_file_path = f"temp/{scan.filename}"
        logger.info(f"Saving uploaded file to {temp_file_path}")
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(scan.file, buffer)
        
        # Determine if this is a generation or comprehension task
        is_generation = taskType in ["enhance", "ct2mri", "mri2ct"]
        
        # Set up the output path for generation tasks
        output_path = f"results/{os.path.splitext(scan.filename)[0]}_{taskType}.jpg" if is_generation else None
        
        # Base command with memory optimizations
        base_cmd = [
            "--dtype", "FP16",
            "--vq_idx_nums", "8192",
            "--instruct_template", "phi3_instruct",
            "--vit_path", "models/ViT",
            "--offload_folder", "offload",
            "--img_path", temp_file_path,
        ]
        
        # Add memory limit if available
        if memory_limit:
            base_cmd.extend(["--max_memory", str(memory_limit)])
        
        # Prepare the command based on task type
        if is_generation:
            logger.info("Preparing generation task command")
            cmd = [
                "python", "gen_infer.py",
                "--model_name_or_path", "microsoft/Phi-3-mini-4k-instruct",
                "--hlora_r", "256",
                "--hlora_alpha", "512",
                "--hlora_nums", "4",
                "--hlora_path", f"models/HealthGPT-M3/gen_hlora_weights.bin",
                "--fusion_layer_path", f"models/HealthGPT-M3/fusion_layer_weights.bin",
                "--vqgan_path", "taming_transformers/ckpt",
                "--question", f"Please {taskType} this medical image.",
                "--save_path", output_path
            ] + base_cmd
        else:
            logger.info("Preparing comprehension task command")
            cmd = [
                "python", "com_infer.py",
                "--model_name_or_path", "microsoft/Phi-3-mini-4k-instruct",
                "--hlora_r", "64",
                "--hlora_alpha", "128",
                "--hlora_nums", "4",
                "--hlora_path", f"models/HealthGPT-M3/com_hlora_weights.bin",
                "--fusion_layer_path", f"models/HealthGPT-M3/fusion_layer_weights.bin",
                "--question", f"Please provide a detailed diagnostic report for this medical image."
            ] + base_cmd
        
        logger.info("Starting inference process")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        stdout, stderr = process.communicate(timeout=300)  # 5-minute timeout
        
        if process.returncode != 0:
            logger.error(f"Inference failed: {stderr}")
            raise HTTPException(status_code=500, detail="Inference failed")
        
        logger.info("Inference completed successfully")
        
        # Return appropriate response based on task type
        if is_generation:
            if not os.path.exists(output_path):
                raise HTTPException(status_code=500, detail="Generated image not found")
            return FileResponse(output_path, media_type="image/jpeg")
        else:
            # Extract the response from stdout
            response_lines = [line for line in stdout.split('\n') if line.strip()]
            if not response_lines:
                raise HTTPException(status_code=500, detail="No response generated")
            return {"result": response_lines[-1].strip()}
        
    except subprocess.TimeoutExpired:
        logger.error("Inference timed out")
        return JSONResponse(
            status_code=504,
            content={"detail": "Processing timed out. Please try again."}
        )
    except Exception as e:
        logger.error(f"Error processing scan: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": str(e)}
        )
    finally:
        # Clean up
        cleanup_memory()
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except:
                pass
        
        logger.info(f"Request completed in {time.time() - start_time:.2f} seconds")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "timestamp": time.time()}

@app.get("/api/test")
async def test_endpoint():
    """Simple test endpoint"""
    return {"message": "API server is working correctly"}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max_memory_mb", type=int, help="Maximum memory to use in MB")
    args = parser.parse_args()
    
    if args.max_memory_mb:
        os.environ["MAX_MEMORY"] = str(args.max_memory_mb)
    
    uvicorn.run(app, host=args.host, port=args.port) 
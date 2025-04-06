import os
import shutil
from huggingface_hub import hf_hub_download, list_repo_files

def download_and_copy(repo_id, filename, target_dir):
    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)
    print(f"Downloading {filename} from {repo_id} ...")
    try:
        # Download the file from the Hugging Face Hub
        cached_path = hf_hub_download(repo_id=repo_id, filename=filename)
        # Construct the destination path
        target_path = os.path.join(target_dir, filename)
        # Copy the file from the cache to your target directory
        shutil.copy(cached_path, target_path)
        print(f"Copied {filename} to {target_path}\n")
    except Exception as e:
        print(f"Failed to download {filename}: {e}\n")

def main():
    # Create directories for different model components
    os.makedirs("models", exist_ok=True)
    os.makedirs("models/HealthGPT-XL32", exist_ok=True)
    os.makedirs("models/HealthGPT-M3", exist_ok=True)
    os.makedirs("models/ViT", exist_ok=True)
    os.makedirs("models/VQGAN", exist_ok=True)
    
    # Download ViT model (required for all HealthGPT models)
    print("Downloading ViT model...")
    vit_repo = "openai/clip-vit-large-patch14-336"
    vit_files = list_repo_files(vit_repo)
    for filename in vit_files:
        download_and_copy(vit_repo, filename, "models/ViT")
    
    # Download HealthGPT-XL32 model (based on Qwen2.5-32B-Instruct)
    print("Downloading HealthGPT-XL32 model...")
    xl32_repo = "lintw/HealthGPT-XL32"
    try:
        xl32_files = list_repo_files(xl32_repo)
        for filename in xl32_files:
            download_and_copy(xl32_repo, filename, "models/HealthGPT-XL32")
    except Exception as e:
        print(f"Error accessing HealthGPT-XL32 repository: {e}")
        print("Note: HealthGPT-XL32 weights might not be fully available yet.")
    
    # Download HealthGPT-M3 model
    print("Downloading HealthGPT-M3 model...")
    m3_repo = "lintw/HealthGPT-M3"
    try:
        m3_files = list_repo_files(m3_repo)
        for filename in m3_files:
            download_and_copy(m3_repo, filename, "models/HealthGPT-M3")
    except Exception as e:
        print(f"Error accessing HealthGPT-M3 repository: {e}")
    
    # Download VQGAN model for image generation
    print("Downloading VQGAN model...")
    vqgan_repo = "CompVis/taming-transformers"
    try:
        vqgan_files = list_repo_files(vqgan_repo)
        for filename in vqgan_files:
            if filename.endswith(".ckpt") or filename.endswith(".yaml"):
                download_and_copy(vqgan_repo, filename, "models/VQGAN")
    except Exception as e:
        print(f"Error accessing VQGAN repository: {e}")
        print("Note: You may need to manually download VQGAN weights from the official source.")
    
    print("\nDownload process completed. Please check the 'models' directory for downloaded files.")
    print("Note: Some models might require manual download if they're not available through the Hugging Face Hub.")

if __name__ == "__main__":
    main() 
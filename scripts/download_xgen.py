import torch
from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoImageProcessor
from pathlib import Path
import json
import os

def load_config():
    config_path = "../config.json"
    with open(config_path, 'r') as f:
        return json.load(f)

def download_xgen_mm():
    """
    Downloads the XGen-MM model and processor from Hugging Face
    """
    config = load_config()
    model_path = Path(config['paths']['models']['pretrained']) / "xgen-mm"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("Downloading XGen-MM model and components...")
    
    model_name_or_path = "Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5"
    
    try:
        # Set HF_HUB_ENABLE_HF_TRANSFER for faster downloads
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        
        # First, download the model which should set up the custom code
        print("Downloading model...")
        model = AutoModelForVision2Seq.from_pretrained(
            model_name_or_path, 
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map=None  # Don't load onto GPU yet
        )
        
        # Now download tokenizer
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, 
            trust_remote_code=True
        )
        
        # Update special tokens
        tokenizer = model.update_special_tokens(tokenizer)
        
        # Now download image processor
        print("Downloading image processor...")
        image_processor = AutoImageProcessor.from_pretrained(
            model_name_or_path, 
            trust_remote_code=True
        )
        
        # Save all components
        print(f"Saving components to {model_path}...")
        model.save_pretrained(str(model_path))
        tokenizer.save_pretrained(str(model_path))
        image_processor.save_pretrained(str(model_path))
        
        print(f"Model and processors saved to {model_path}")
        return model_path
        
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        print("Full error:", e.__class__.__name__)
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    download_xgen_mm()

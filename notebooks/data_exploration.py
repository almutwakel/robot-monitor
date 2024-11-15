import torch
from pathlib import Path
import json
from PIL import Image
import PIL.Image
import requests
from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoImageProcessor
import matplotlib.pyplot as plt
import os

def load_config():
    """Load configuration from config.json"""
    config_path = Path(__file__).parents[1] / "config.json"
    with open(config_path, 'r') as f:
        return json.load(f)

def load_model_and_processors():
    """Load the XGen-MM model and processors"""
    config = load_config()
    model_path = Path(config['paths']['models']['pretrained']) / "xgen-mm"
    original_model_id = "Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5"
    
    print("Loading model and processors...")
    
    # For model, we can use local path if it exists
    if os.path.isdir(model_path):
        print("Loading model from local path...")
        model = AutoModelForVision2Seq.from_pretrained(
            str(model_path), 
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
    else:
        print("Downloading model from HuggingFace...")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model = AutoModelForVision2Seq.from_pretrained(
            original_model_id, 
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        model.save_pretrained(str(model_path))
    
    # Always use the original model ID for tokenizer and processor
    print("Loading tokenizer and processor...")
    tokenizer = AutoTokenizer.from_pretrained(
        original_model_id,  # Use original model ID, not local path
        trust_remote_code=True,
        use_fast=False,
        legacy=False
    )
    
    image_processor = AutoImageProcessor.from_pretrained(
        original_model_id,  # Use original model ID, not local path
        trust_remote_code=True
    )
    
    tokenizer = model.update_special_tokens(tokenizer)
    
    return model, tokenizer, image_processor

def load_demo_image():
    """Load a sample image for testing"""
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    return image

def visualize_image(image):
    """Display the input image"""
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def run_inference(model, tokenizer, image_processor, image, prompt="What do you see in this image?"):
    """Run inference with the model on the given image"""
    # Process the inputs
    inputs = image_processor(images=image, return_tensors="pt")
    text_inputs = tokenizer(prompt, return_tensors="pt")
    
    inputs.update(text_inputs)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate response
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
        num_beams=5,
    )
    
    # Decode and return the response
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

def main():
    # Load model and processors
    model, tokenizer, image_processor = load_model_and_processors()
    
    # Load and display test image
    image = load_demo_image()
    print("Displaying test image...")
    visualize_image(image)
    
    # Run inference
    print("\nRunning inference...")
    response = run_inference(model, tokenizer, image_processor, image)
    print("\nModel's response:")
    print(response)

if __name__ == "__main__":
    main()

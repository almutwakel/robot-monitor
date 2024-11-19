from abc import ABC, abstractmethod
import torch
from typing import List, Dict, Any, Union
from pathlib import Path
import json
import re
import base64
from io import BytesIO
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModelForVision2Seq
import os
from src.utils.logging_utils import setup_logger, log_dict
from datetime import datetime

# Create single logger instance for the module
logger = setup_logger()

class VisionLanguageModel(ABC):
    """Base class for vision-language models"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    @abstractmethod
    def predict_success(self, image: torch.Tensor, task_description: str) -> Dict[str, float]:
        """
        Predict success probability for a task given an image
        Args:
            image: Tensor of shape [C, H, W] or [C, H*num_frames, W]
            task_description: String describing the task
        Returns:
            Dict containing success probability and confidence
        """
        pass
    
    @abstractmethod
    def get_attention_maps(self) -> Union[torch.Tensor, None]:
        """
        Get attention maps from the last prediction
        Returns:
            Attention tensor or None if not available
        """
        pass

class OpenAIVisionModel(VisionLanguageModel):
    """OpenAI's GPT-4V API implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Load API key
        try:
            key_path = Path(config['api_keys']['openai_key_path'])
            with open(key_path, 'r') as f:
                self.api_key = f.read().strip()
        except Exception as e:
            logger.error(f"Error loading OpenAI API key: {str(e)}")
            raise
            
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.model = "gpt-4o-mini"
        self.last_attention = None
    
    def _encode_image(self, image: torch.Tensor) -> str:
        """Convert tensor to base64 string"""
        # Convert to PIL Image
        image = image.permute(1, 2, 0)  # [H, W, C]
        image = (image * 255).cpu().numpy().astype('uint8')
        pil_image = Image.fromarray(image)
        
        # Convert to base64
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def predict_success(self, image: torch.Tensor, task_description: str) -> Dict[str, Any]:
        try:
            # Prepare the API request
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # Log API key presence (not the actual key)
            logger.info(f"API key loaded and has length: {len(self.api_key)}")
            
            # Encode image
            base64_image = self._encode_image(image)
            
            # Construct prompt
            prompt = f"""Given the task "{task_description}", analyze the image and determine if the task was completed successfully.
                        Respond with success boolean (true or false) and your confidence in this assessment based on whether you have enough information to determine success and your ability to reason about the task (very low, low, medium, high, very high).
                        Add an explanation for your assessment. Use the following schema: {{"success": boolean, "confidence": "very low" | "low" | "medium" | "high" | "very high", "explanation": string}}"""
            
            # Prepare request payload according to latest API spec
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "low"  # Use low detail to reduce token usage
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 300
            }
            
            # Log the request
            log_dict(logger, {
                "timestamp": datetime.now().isoformat(),
                "model": self.model,
                "task": task_description,
                "prompt": prompt,
                "request": payload
            }, prefix="Request: ")
            
            # Make API request
            logger.info(f"Making request to: {self.api_url}")
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload
            )
            
            # Log the response
            log_dict(logger, {
                "timestamp": datetime.now().isoformat(),
                "status_code": response.status_code,
                "response": response.json() if response.status_code == 200 else response.text
            }, prefix="Response: ")
            
            if response.status_code != 200:
                logger.error(f"API Response Status: {response.status_code}")
                logger.error(f"API Response Text: {response.text}")
                
            response.raise_for_status()
            result = response.json()
            
            # Parse response
            content = result['choices'][0]['message']['content']
            
            # Extract JSON from content
            try:
                result = json.loads(content)
                
                # Validate the response format
                if not isinstance(result.get('success'), bool):
                    raise ValueError("Success must be a boolean")
                    
                if result.get('confidence') not in ["very low", "low", "medium", "high", "very high"]:
                    raise ValueError("Confidence must be one of: very low, low, medium, high, very high")
                    
                if not isinstance(result.get('explanation'), str):
                    raise ValueError("Explanation must be a string")
                
                return {
                    **result,
                    "valid": True
                }
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Error parsing model output: {str(e)}. Using fallback parsing.")
                
                # Fallback parsing
                success_match = re.search(r'success["\s:]+(\w+)', content.lower())
                confidence_match = re.search(r'confidence["\s:]+(\w+(?:\s+\w+)?)', content.lower())
                
                if not (success_match and confidence_match):
                    return {
                        "success": None,
                        "confidence": None,
                        "explanation": content,
                        "valid": False
                    }
                
                success_str = success_match.group(1)
                conf_str = confidence_match.group(1).strip()
                
                if (success_str.strip() in ['true', 'yes', '1'] or 
                    success_str.strip() in ['false', 'no', '0']) and \
                   conf_str in ["very low", "low", "medium", "high", "very high"]:
                    
                    return {
                        "success": success_str.strip() in ['true', 'yes', '1'],
                        "confidence": conf_str,
                        "explanation": content,
                        "valid": True
                    }
                
                return {
                    "success": None,
                    "confidence": None,
                    "explanation": content,
                    "valid": False
                }
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            logger.error(f"Response status code: {e.response.status_code if hasattr(e, 'response') else 'N/A'}")
            logger.error(f"Response text: {e.response.text if hasattr(e, 'response') else 'N/A'}")
            raise
    
    def get_attention_maps(self) -> None:
        """OpenAI API doesn't provide attention maps"""
        return None

class XgenMMModel(VisionLanguageModel):
    """Salesforce's Xgen-MM model implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Load model and processor
        try:
            model_path = Path(config['paths']['models']['pretrained']) / "xgen-7b-8k-inst-mm"
            self.processor = AutoProcessor.from_pretrained(model_path)
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.model.eval()
        except Exception as e:
            logger.error(f"Error loading Xgen-MM model: {str(e)}")
            raise
            
        self.last_attention = None
    
    def predict_success(self, image: torch.Tensor, task_description: str) -> Dict[str, Any]:
        try:
            # Prepare prompt
            prompt = f"""Given the task "{task_description}", analyze the image and determine if the task was completed successfully.
                        Respond with success boolean (true or false) and your confidence in this assessment based on whether you have enough information to determine success and your ability to reason about the task (very low, low, medium, high, very high).
                        Add an explanation for your assessment. Use the following schema: {{"success": boolean, "confidence": "very low" | "low" | "medium" | "high" | "very high", "explanation": string}}"""
            
            # Process inputs
            inputs = self.processor(
                images=image.permute(1, 2, 0).cpu().numpy(),
                text=prompt,
                return_tensors="pt"
            ).to(self.model.device)
            
            # Generate prediction
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    output_attentions=True,
                    return_dict_in_generate=True
                )
                
                # Store attention maps
                self.last_attention = outputs.attentions
                
                # Decode prediction
                prediction = self.processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
                
                # Parse JSON from prediction
                import json
                try:
                    result = json.loads(prediction)
                    
                    # Validate the response format
                    if not isinstance(result.get('success'), bool):
                        raise ValueError("Success must be a boolean")
                        
                    if result.get('confidence') not in ["very low", "low", "medium", "high", "very high"]:
                        raise ValueError("Confidence must be one of: very low, low, medium, high, very high")
                        
                    if not isinstance(result.get('explanation'), str):
                        raise ValueError("Explanation must be a string")
                    
                    return {
                        **result,
                        "valid": True
                    }
                    
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Error parsing model output: {str(e)}. Using fallback parsing.")
                    
                    # Fallback parsing
                    success_match = re.search(r'success["\s:]+(\w+)', prediction.lower())
                    confidence_match = re.search(r'confidence["\s:]+(\w+(?:\s+\w+)?)', prediction.lower())
                    
                    if not (success_match and confidence_match):
                        return {
                            "success": None,
                            "confidence": None,
                            "explanation": prediction,
                            "valid": False
                        }
                    
                    success_str = success_match.group(1)
                    conf_str = confidence_match.group(1).strip()
                    
                    if (success_str.strip() in ['true', 'yes', '1'] or 
                        success_str.strip() in ['false', 'no', '0']) and \
                       conf_str in ["very low", "low", "medium", "high", "very high"]:
                        
                        return {
                            "success": success_str.strip() in ['true', 'yes', '1'],
                            "confidence": conf_str,
                            "explanation": prediction,
                            "valid": True
                        }
                    
                    return {
                        "success": None,
                        "confidence": None,
                        "explanation": prediction,
                        "valid": False
                    }
                
        except Exception as e:
            logger.error(f"Error in Xgen-MM prediction: {str(e)}")
            raise
    
    def get_attention_maps(self) -> Union[torch.Tensor, None]:
        """Return stored attention maps from last prediction"""
        return self.last_attention

def get_vision_model(config: Dict[str, Any]) -> VisionLanguageModel:
    """Factory function to create vision model based on config"""
    model_type = config['model_configs'].get('model_type', 'openai')
    
    if model_type == 'openai':
        return OpenAIVisionModel(config)
    elif model_type == 'xgen':
        return XgenMMModel(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def load_config():
    """Load configuration from config.json"""
    try:
        config_path = Path(__file__).parents[1] / "config.json"
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        raise

# Example usage
if __name__ == "__main__":
    config = load_config()
    
    # Create model
    model = get_vision_model(config)
    
    # Example prediction
    dummy_image = torch.randn(3, 224, 224)
    task = "pick up the blue cube"
    
    try:
        prediction = model.predict_success(dummy_image, task)
        print(f"Prediction: {prediction}")
        
        attention = model.get_attention_maps()
        if attention is not None:
            print(f"Attention shape: {attention[0].shape}")
    except Exception as e:
        logger.error(f"Error in example: {str(e)}")

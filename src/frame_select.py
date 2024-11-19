from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import json
from typing import List, Dict, Any
from datetime import datetime
from src.utils.logging_utils import setup_logger, log_dict

# Create single logger instance for the module
logger = setup_logger()

class FrameSelector(ABC):
    """Abstract base class for frame selection strategies"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.num_frames = config['frame_sampling'].get('num_frames', 2)
    
    @abstractmethod
    def select_frames(self, frames: List[torch.Tensor], task_description: str = None) -> torch.Tensor:
        """
        Select and concatenate frames from a trajectory
        Args:
            frames: List of frame tensors [C, H, W]
            task_description: Optional task description for VLM-guided selection
        Returns:
            Concatenated tensor of selected frames [C, H*num_frames, W]
        """
        pass

class EndpointInterpolationSelector(FrameSelector):
    """Selects first, last, and equally spaced frames in between"""
    
    def select_frames(self, frames: List[torch.Tensor], task_description: str = None) -> torch.Tensor:
        num_total_frames = len(frames)
        
        if num_total_frames == 0:
            raise ValueError("Empty frame list provided")
            
        if self.num_frames > num_total_frames:
            logger.warning(f"Requested {self.num_frames} frames but only {num_total_frames} available")
            self.num_frames = num_total_frames
            
        if self.num_frames == 1:
            # Use last frame only
            selected = [frames[-1]]
        
        elif self.num_frames == 2:
            # Use first and last frames
            selected = [frames[0], frames[-1]]
            
        else:
            # Include first, last, and equally spaced frames in between
            indices = np.linspace(0, num_total_frames-1, self.num_frames, dtype=int)
            selected = [frames[i] for i in indices]
        
        # Concatenate frames vertically
        return torch.cat(selected, dim=1)

class MaxDifferenceSelector(FrameSelector):
    """Selects frames that are maximally different from each other"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.metric = config['frame_sampling'].get('max_difference_metric', 'mse')
        
    def compute_difference(self, frame1: torch.Tensor, frame2: torch.Tensor) -> float:
        """Compute difference between two frames using specified metric"""
        if self.metric == 'mse':
            return F.mse_loss(frame1, frame2).item()
        elif self.metric == 'mae':
            return F.l1_loss(frame1, frame2).item()
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")
    
    def select_frames(self, frames: List[torch.Tensor], task_description: str = None) -> torch.Tensor:
        if len(frames) == 0:
            raise ValueError("Empty frame list provided")
            
        if self.num_frames > len(frames):
            logger.warning(f"Requested {self.num_frames} frames but only {len(frames)} available")
            self.num_frames = len(frames)
        
        # Always include the last frame
        selected_indices = {len(frames) - 1}
        selected = [frames[-1]]
        
        # Greedily select frames that are most different from already selected ones
        while len(selected) < self.num_frames:
            max_min_diff = -1
            best_idx = -1
            
            # For each candidate frame
            for i in range(len(frames)):
                if i in selected_indices:
                    continue
                    
                # Compute minimum difference to any selected frame
                min_diff = float('inf')
                for selected_frame in selected:
                    diff = self.compute_difference(frames[i], selected_frame)
                    min_diff = min(min_diff, diff)
                
                # Update best candidate if this frame has larger minimum difference
                if min_diff > max_min_diff:
                    max_min_diff = min_diff
                    best_idx = i
            
            selected_indices.add(best_idx)
            selected.append(frames[best_idx])
        
        # Sort frames by temporal order
        selected = [frames[i] for i in sorted(selected_indices)]
        return torch.cat(selected, dim=1)

class VLMGuidedSelector(FrameSelector):
    """VLM-guided frame selection"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = config['frame_sampling'].get('model', 'mini-clip')
        logger.warning("VLMGuidedSelector is a placeholder and not yet implemented")
    
    def select_frames(self, frames: List[torch.Tensor], task_description: str = None) -> torch.Tensor:
        # For now, fall back to endpoint interpolation
        selector = EndpointInterpolationSelector(self.config)
        return selector.select_frames(frames, task_description)

def get_frame_selector(config: Dict[str, Any]) -> FrameSelector:
    """Factory function to create frame selector based on config"""
    selector_type = config['frame_sampling'].get('type', 'endpoint-interpolation')
    
    if selector_type == 'endpoint-interpolation':
        return EndpointInterpolationSelector(config)
    elif selector_type == 'max-difference':
        return MaxDifferenceSelector(config)
    elif selector_type == 'vlm-guided':
        return VLMGuidedSelector(config)
    else:
        raise ValueError(f"Unknown frame selector type: {selector_type}")

def load_config():
    """Load configuration from config.json"""
    try:
        config_path = Path(__file__).parents[1] / "config.json"
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        raise

# Example usage:
if __name__ == "__main__":
    # Load config
    config = load_config()
    frame_sampling_config = config.get('frame_sampling', {})
    
    # Create selector
    selector = get_frame_selector(frame_sampling_config)
    
    # Example frames (normally would come from dataloader)
    dummy_frames = [torch.randn(3, 224, 224) for _ in range(10)]
    
    # Select frames
    try:
        concatenated = selector.select_frames(dummy_frames)
        print(f"Selected frame shape: {concatenated.shape}")
    except Exception as e:
        logger.error(f"Error selecting frames: {str(e)}")

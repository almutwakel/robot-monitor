import torch
from torch.utils.data import Dataset, DataLoader
import tensorflow as tf
import numpy as np
from pathlib import Path
import json
import glob
import os
import itertools
from collections import defaultdict
import logging
from tqdm import tqdm
from typing import Dict, Any
from datetime import datetime
from src.utils.logging_utils import setup_logger, log_dict

# Create single logger instance for the module
logger = setup_logger()

class ManipulationDataset(Dataset):
    def __init__(self, successes_paths, failures_paths, success_ratio=0.5, num_samples=None, debug=False):
        """
        Args:
            successes_paths: List of paths containing success trajectories
            failures_paths: List of paths containing failure trajectories
            success_ratio: Ratio of success samples to keep
            num_samples: Total number of samples to keep (if None, keep all)
            debug: If True, enables verbose logging and saves first batch of images
        """
        self.success_ratio = success_ratio
        self.debug = debug
        self.num_samples = num_samples
        logger.info(f"Initializing ManipulationDataset with {len(successes_paths)} success paths and {len(failures_paths)} failure paths")
        
        if self.debug:
            logger.setLevel(logging.DEBUG)
        
        # Define features schema for TFRecord parsing
        self.features = {
            'present/image/encoded': tf.io.FixedLenFeature([], tf.string),
            'episode_id': tf.io.FixedLenFeature([1], tf.string),
            'subtask_id': tf.io.FixedLenFeature([1], tf.int64),
            'subtask_name': tf.io.FixedLenFeature([1], tf.string),
            'sentence_embedding': tf.io.FixedLenFeature([512], tf.float32),
            'sequence_length': tf.io.FixedLenFeature([1], tf.int64),
            'present/timestep_count': tf.io.FixedLenFeature([1], tf.float32),
            'present/xyz': tf.io.FixedLenFeature([3], tf.float32),
            'present/axis_angle': tf.io.FixedLenFeature([3], tf.float32),
        }
        
        self.trajectories = []
        self.labels = []
        
        try:
            # Calculate target numbers for success and failure samples
            if num_samples is not None:
                target_success = int(num_samples * success_ratio)
                target_failure = num_samples - target_success
            else:
                target_success = target_failure = float('inf')
            
            # Process success files
            logger.info("Processing success trajectories...")
            success_count = 0
            for path in successes_paths:
                if success_count >= target_success:
                    break
                success_files = glob.glob(os.path.join(path, '*'))
                if not success_files:
                    logger.warning(f"No files found in success path: {path}")
                success_count = self._process_files(success_files, is_success=True, 
                                                  target_count=target_success, current_count=success_count)
            
            # Process failure files
            logger.info("Processing failure trajectories...")
            failure_count = 0
            for path in failures_paths:
                if failure_count >= target_failure:
                    break
                failure_files = glob.glob(os.path.join(path, '*'))
                if not failure_files:
                    logger.warning(f"No files found in failure path: {path}")
                failure_count = self._process_files(failure_files, is_success=False, 
                                                  target_count=target_failure, current_count=failure_count)
            
            logger.info(f"Loaded {len(self.trajectories)} total trajectories")
            logger.info(f"Success trajectories: {sum(self.labels)}")
            logger.info(f"Failure trajectories: {len(self.labels) - sum(self.labels)}")
            
        except Exception as e:
            logger.error(f"Error during dataset initialization: {str(e)}", exc_info=True)
            raise

    def _process_files(self, files, is_success, target_count, current_count):
        """
        Process TFRecord files and group frames by episode.
        Returns the updated count of processed trajectories.
        """
        logger.debug(f"Processing {'success' if is_success else 'failure'} files: {len(files)} files found")
        
        current_episode_frames = []
        current_episode_id = None
        current_episode_data = {}
        
        for file in files:  # No need for tqdm here as we'll process very few examples
            try:
                # Read dataset and take only what we might need (assuming each episode has ~50 frames)
                # This ensures we read minimal data from disk
                dataset = tf.data.TFRecordDataset(file).take((target_count - current_count) * 50)
                
                for serialized_example in dataset:
                    if current_count >= target_count:
                        return current_count
                    
                    example = self._parse_tfrecord(serialized_example)
                    episode_id = example['episode_id'][0].decode('utf-8')
                    
                    # If we're starting a new episode
                    if current_episode_id is not None and episode_id != current_episode_id:
                        # Process the completed episode
                        if current_episode_frames:
                            sorted_frames = sorted(current_episode_frames, key=lambda x: x[0])
                            trajectory = {
                                'frames': [f[1] for f in sorted_frames],
                                'sentence_embedding': current_episode_data['sentence_embedding'],
                                'subtask_name': current_episode_data['subtask_name'],
                            }
                            self.trajectories.append(trajectory)
                            self.labels.append(1 if is_success else 0)
                            current_count += 1
                            
                            if self.debug and len(self.trajectories) == 1:
                                self._save_debug_info(trajectory, current_episode_id)
                            
                            # Clear for next episode
                            current_episode_frames = []
                            current_episode_data = {}
                            
                            if current_count >= target_count:
                                return current_count
                    
                    # Store frame data
                    current_episode_id = episode_id
                    timestep = int(example['present/timestep_count'][0])
                    current_episode_frames.append((timestep, example))
                    
                    if not current_episode_data:
                        current_episode_data = {
                            'sentence_embedding': example['sentence_embedding'],
                            'subtask_name': example['subtask_name'][0].decode('utf-8'),
                        }
                
                # Process the last episode if needed
                if current_episode_frames and current_count < target_count:
                    sorted_frames = sorted(current_episode_frames, key=lambda x: x[0])
                    trajectory = {
                        'frames': [f[1] for f in sorted_frames],
                        'sentence_embedding': current_episode_data['sentence_embedding'],
                        'subtask_name': current_episode_data['subtask_name'],
                    }
                    self.trajectories.append(trajectory)
                    self.labels.append(1 if is_success else 0)
                    current_count += 1
                    
                    if self.debug and len(self.trajectories) == 1:
                        self._save_debug_info(trajectory, current_episode_id)
                    
                if current_count >= target_count:
                    return current_count
                    
            except Exception as e:
                logger.error(f"Error processing file {file}: {str(e)}")
                continue
        
        return current_count

    def _save_debug_info(self, trajectory, episode_id):
        """Save debug information for the first trajectory"""
        try:
            import matplotlib.pyplot as plt
            debug_dir = Path("debug_output")
            debug_dir.mkdir(exist_ok=True)
            
            # Save first frame
            # first_frame = trajectory['frames'][0]['present/image/encoded'].numpy()
            # plt.imsave(debug_dir / f"episode_{episode_id}_frame_0.png", first_frame)
            
            # Save trajectory info
            with open(debug_dir / f"episode_{episode_id}_info.txt", 'w') as f:
                f.write(f"Subtask: {trajectory['subtask_name']}\n")
                f.write(f"Number of frames: {len(trajectory['frames'])}\n")
                f.write(f"Embedding shape: {trajectory['sentence_embedding'].shape}\n")
            
            logger.debug(f"Saved debug information to {debug_dir}")
        except Exception as e:
            logger.error(f"Error saving debug info: {str(e)}")

    def _parse_tfrecord(self, serialized_example):
        """Parse TFRecord example"""
        try:
            example = tf.io.parse_single_example(serialized_example, self.features)
            # Convert image from tensor to numpy before decoding
            example['present/image/encoded'] = tf.io.decode_jpeg(example['present/image/encoded']).numpy()
            # Convert other string tensors to numpy
            example['episode_id'] = example['episode_id'].numpy()
            example['subtask_name'] = example['subtask_name'].numpy()
            example['sentence_embedding'] = example['sentence_embedding'].numpy()
            return example
        except Exception as e:
            logger.error(f"Error parsing TFRecord: {str(e)}")
            raise

    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        try:
            trajectory = self.trajectories[idx]
            
            # Convert each frame to a tensor, but keep as list
            frames = [torch.from_numpy(frame['present/image/encoded']).permute(2, 0, 1) for frame in trajectory['frames']]  # List of [C, H, W] tensors
            label = torch.tensor(self.labels[idx])
            
            # Log item access
            log_dict(logger, {
                "timestamp": datetime.now().isoformat(),
                "index": idx,
                "trajectory_id": idx,
                "subtask": trajectory['subtask_name'],
                "num_frames": len(frames),
                "success": bool(label)
            }, prefix="Dataset Access: ")
            
            return {
                'frames': frames,  # List of frame tensors
                'label': label,
                'sentence_embedding': torch.from_numpy(trajectory['sentence_embedding']),
                'subtask_name': trajectory['subtask_name'],
                'num_frames': len(frames)
            }
        except Exception as e:
            logger.error(f"Error getting item {idx}: {str(e)}")
            raise

def collate_fn(batch):
    """
    Custom collate function that keeps the original frame sequences as lists.
    No padding is applied - each sequence maintains its original length.
    """
    return {
        'frames': [item['frames'] for item in batch],  # List of variable-length sequences
        'label': torch.stack([item['label'] for item in batch]),
        'sentence_embedding': torch.stack([item['sentence_embedding'] for item in batch]),
        'subtask_name': [item['subtask_name'] for item in batch],
        'num_frames': torch.tensor([item['num_frames'] for item in batch])
    }

def get_dataloader(debug=False):
    """
    Returns the pytorch dataloader for the dataset.
    Args:
        debug: If True, enables verbose logging and saves debug information
    """
    try:
        config = load_config()
        
        successes_paths = [Path(path) for path in config['paths']['data']['successes']]
        failures_paths = [Path(path) for path in config['paths']['data']['failures']]
        
        success_ratio = config['data_loader'].get('success_ratio', 0.5)
        num_samples = config['data_loader'].get('num_samples', None)
        
        logger.info("Creating dataset...")
        dataset = ManipulationDataset(
            successes_paths, 
            failures_paths,
            success_ratio=success_ratio,
            num_samples=num_samples,
            debug=debug
        )
        print("Dataset length:", len(dataset))

        logger.info("Creating dataloader...")
        dataloader = DataLoader(
            dataset, 
            batch_size=config['experiment']['batch_size'],
            num_workers=config['experiment']['num_workers'],
            shuffle=True,
            collate_fn=collate_fn  # Add custom collate function
        )

        return dataloader
    except Exception as e:
        logger.error(f"Error creating dataloader: {str(e)}", exc_info=True)
        raise

def load_config():
    """Load configuration from config.json"""
    try:
        config_path = Path(__file__).parents[1] / "config.json"
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        raise
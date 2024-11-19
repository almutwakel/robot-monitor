import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import seaborn as sns
from datetime import datetime
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

from src.dataloader import get_dataloader
from src.utils.logging_utils import setup_logger
from src.vision import get_vision_model
from src.frame_select import get_frame_selector

# Create logger
logger = setup_logger()

class ModelEvaluator:
    def __init__(self, config_path=None):
        """Initialize the evaluator with config and create necessary directories"""
        self.config = self._load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create results directory
        self.results_dir = Path(self.config.get('paths', {}).get('results', 'results'))
        self.results_dir.mkdir(exist_ok=True)
        
        # Create timestamped directory for this evaluation run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.eval_dir = self.results_dir / self.timestamp
        self.eval_dir.mkdir(exist_ok=True)
        
        logger.info(f"Evaluation results will be saved to {self.eval_dir}")

    def _load_config(self, config_path=None):
        """Load configuration from file"""
        if config_path is None:
            config_path = Path(__file__).parents[1] / "config.json"
        
        with open(config_path, 'r') as f:
            return json.load(f)

    def evaluate(self):
        """Run evaluation and save results"""
        # Get model, frame selector and dataloader
        model = get_vision_model(self.config)
        frame_selector = get_frame_selector(self.config)
        dataloader = get_dataloader(self.config, debug=False)
        iterator = iter(dataloader)
        
        results = []
        
        # Evaluate each trajectory
        for i in range(self.config['data_loader']['num_samples']):
            try:
                batch = next(iterator)
                
                # Process each trajectory in the batch
                for traj_idx in range(len(batch['frames'])):
                    # Get frames and task info for this trajectory
                    frames = batch['frames'][traj_idx]  # List of frame tensors
                    task_description = batch['subtask_name'][traj_idx]
                    
                    # Select frames using frame selector
                    selected_frames = frame_selector.select_frames(frames, task_description)
                    
                    # Get predictions using vision model
                    prediction = model.predict_success(selected_frames, task_description)
                    
                    # Store results
                    results.append({
                        'traj_idx': i,
                        'true_label': batch['label'][traj_idx].item(),
                        'predicted': prediction['success'],
                        'confidence': prediction['confidence'],
                        'explanation': prediction.get('explanation', ''),
                        'subtask_name': task_description,
                        'num_frames': batch['num_frames'][traj_idx].item(),
                        'num_selected_frames': len(frames)
                    })
                    
            except Exception as e:
                logger.error(f"Error processing trajectory {i}: {str(e)}", exc_info=True)
                continue

        # Convert results to DataFrame and analyze
        df = pd.DataFrame(results)
        self._save_and_analyze_results(df)

    def _save_and_analyze_results(self, df):
        """Save results and create visualizations"""
        # Save raw results
        df.to_csv(self.eval_dir / 'evaluation_results.csv', index=False)
        
        # Generate visualizations and analysis
        self._plot_confusion_matrix(df)
        self._analyze_confidence(df)
        self._analyze_subtasks(df)
        self._analyze_frame_selection(df)
        self._save_detailed_report(df)

    def _analyze_frame_selection(self, df):
        """Analyze frame selection statistics"""
        plt.figure(figsize=(10, 6))
        plt.scatter(df['num_frames'], df['num_selected_frames'])
        plt.xlabel('Total Frames in Trajectory')
        plt.ylabel('Number of Selected Frames')
        plt.title('Frame Selection Analysis')
        plt.savefig(self.eval_dir / 'frame_selection_analysis.png')
        plt.close()

    def _plot_confusion_matrix(self, df):
        """Create and save confusion matrix plot"""
        cm = confusion_matrix(df['true_label'], df['predicted'])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(self.eval_dir / 'confusion_matrix.png')
        plt.close()

    def _analyze_confidence(self, df):
        """Analyze prediction confidence"""
        # Map confidence strings to numeric values
        confidence_map = {
            'very low': 1,
            'low': 2,
            'medium': 3,
            'high': 4,
            'very high': 5
        }
        
        # Convert confidence strings to numeric values
        df['confidence_numeric'] = df['confidence'].map(confidence_map)
        
        # Plot confidence distribution
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='true_label', y='confidence_numeric')
        plt.title('Confidence Distribution by True Label')
        plt.ylabel('Confidence Level')
        plt.yticks(range(1,6), confidence_map.keys())
        plt.savefig(self.eval_dir / 'confidence_distribution.png')
        plt.close()
        
        # Calculate accuracy by confidence level - explicitly select columns
        accuracy_by_confidence = df.groupby('confidence')[['predicted', 'true_label']].apply(
            lambda x: (x['predicted'] == x['true_label']).mean()
        ).reindex(['very low', 'low', 'medium', 'high', 'very high'])
        
        # Plot accuracy vs confidence
        plt.figure(figsize=(10, 6))
        accuracy_by_confidence.plot(kind='bar')
        plt.title('Accuracy by Confidence Level')
        plt.ylabel('Accuracy')
        plt.xlabel('Confidence')
        plt.tight_layout()
        plt.savefig(self.eval_dir / 'accuracy_by_confidence.png')
        plt.close()

    def _analyze_subtasks(self, df):
        """Analyze performance by subtask"""
        # Map confidence strings to numeric values
        confidence_map = {
            'very low': 1,
            'low': 2,
            'medium': 3,
            'high': 4,
            'very high': 5
        }
        df['confidence_numeric'] = df['confidence'].map(confidence_map)
        
        # Explicitly select columns for groupby
        subtask_performance = df.groupby('subtask_name')[['predicted', 'true_label', 'confidence_numeric']].apply(
            lambda x: pd.Series({
                'accuracy': (x['predicted'] == x['true_label']).mean(),
                'avg_confidence_numeric': x['confidence_numeric'].mean(),
                'count': len(x)
            })
        )
        
        # Plot subtask performance
        plt.figure(figsize=(12, 6))
        ax = subtask_performance[['accuracy']].plot(kind='bar')
        plt.title(f'Performance by Subtask (Total samples: {len(df)})')
        
        # Add count annotations on top of each bar
        for i, v in enumerate(subtask_performance['count']):
            ax.text(i, subtask_performance['accuracy'][i], 
                    f'n={v}', 
                    horizontalalignment='center',
                    verticalalignment='bottom')
        
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1.1)  # Set y-axis limit to make room for annotations
        plt.tight_layout()
        plt.savefig(self.eval_dir / 'subtask_performance.png')
        plt.close()

    def _save_detailed_report(self, df):
        """Save detailed analysis report"""
        report = []
        
        # Overall metrics
        accuracy = (df['predicted'] == df['true_label']).mean()
        
        # Map confidence strings to numeric values for averaging
        confidence_map = {
            'very low': 1,
            'low': 2,
            'medium': 3,
            'high': 4,
            'very high': 5
        }
        df['confidence_numeric'] = df['confidence'].map(confidence_map)
        avg_confidence_num = df['confidence_numeric'].mean()
        
        # Map back to closest confidence level
        confidence_levels = list(confidence_map.keys())
        avg_confidence = confidence_levels[round(avg_confidence_num) - 1]
        
        report.append("=== Overall Performance ===")
        report.append(f"Total trajectories: {len(df)}")
        report.append(f"Overall accuracy: {accuracy:.3f}")
        report.append(f"Average confidence: {avg_confidence}")
        report.append("\n=== Classification Report ===")
        report.append(classification_report(df['true_label'], df['predicted'], zero_division=0))
        
        # Per-trajectory analysis
        report.append("\n=== Individual Trajectory Analysis ===")
        for _, row in df.iterrows():
            report.append(
                f"Trajectory {row['traj_idx']} "
                f"(Subtask: {row['subtask_name']}): "
                f"True={row['true_label']}, Pred={row['predicted']}, "
                f"Conf={row['confidence']}"  # Removed .3f format since confidence is a string
            )
        
        # Save report
        with open(self.eval_dir / 'evaluation_report.txt', 'w') as f:
            f.write('\n'.join(report))

def main():
    evaluator = ModelEvaluator()
    evaluator.evaluate()

if __name__ == "__main__":
    main()

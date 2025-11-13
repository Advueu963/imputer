"""GPU-accelerated imputer for image data using PyTorch."""

import torch
import torch.nn.functional as F
from torchvision.io import read_image
from torchvision.models import resnet18, ResNet18_Weights
from typing import Literal
import numpy as np
import urllib.request
import json


class GPUBaselineImputer:
    """GPU-accelerated baseline imputer with mean/median/constant strategies."""
    
    def __init__(
        self, 
        model,
        data: torch.Tensor,
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=32,
        strategy: Literal["mean", "median", "constant"] = "mean",
        constant_value: float = 0.0
    ):
        """Initialize GPU baseline imputer.
        
        Args:
            model: PyTorch model
            data: Background data tensor (N, C, H, W)
            device: Device to run on
            batch_size: Batch size for processing
            strategy: Imputation strategy ('mean', 'median', 'constant')
            constant_value: Value for constant strategy
        """
        self.device = device
        self.batch_size = batch_size
        self.strategy = strategy
        self.constant_value = constant_value
        self.model = model.to(device)
        self.model.eval()
        
        # Calculate baseline values from data
        self.data = data.to(device)
        self._compute_baseline()
    
    def _compute_baseline(self):
        """Compute baseline values based on strategy."""
        if self.strategy == "mean":
            self.baseline = self.data.mean(dim=0, keepdim=True)
        elif self.strategy == "median":
            self.baseline = self.data.median(dim=0, keepdim=True).values
        elif self.strategy == "constant":
            self.baseline = torch.full_like(self.data[0:1], self.constant_value)
    
    def impute_with_coalition(self, x: torch.Tensor, coalition: torch.Tensor):
        """Impute features based on coalition mask.
        
        Args:
            x: Input tensor (B, C, H, W)
            coalition: Boolean tensor (C,) - True=keep, False=impute
            
        Returns:
            Imputed tensor
        """
        x = x.to(self.device)
        coalition = coalition.to(self.device)
        
        # Expand coalition to match tensor dimensions
        imputed = x.clone()
        for c in range(x.shape[1]):
            if not coalition[c]:
                imputed[:, c] = self.baseline[:, c]
        
        return imputed
    
    def predict_batch(self, tensors: torch.Tensor):
        """Predict on batch of tensors.
        
        Args:
            tensors: Input tensor batch (B, C, H, W)
            
        Returns:
            Model predictions
        """
        tensors = tensors.to(self.device)
        with torch.no_grad():
            return F.softmax(self.model(tensors), dim=1)


class GPUImageImputer:
    """GPU-accelerated imputer for images using ResNet18."""
    
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu", batch_size=32):
        """Initialize imputer with device and batch size.
        
        Args:
            device: Device to run on ('cpu' or 'cuda')
            batch_size: Batch size for processing multiple images
        """
        self.device = device
        self.batch_size = batch_size
        
        # Load pre-trained ResNet18
        weights = ResNet18_Weights.DEFAULT
        self.model = resnet18(weights=weights).to(device)
        self.model.eval()
        self.preprocess = weights.transforms()
        
    def impute_single(self, image_path):
        """Impute a single image.
        
        Args:
            image_path: Path to the image
            
        Returns:
            predictions: Model predictions
        """
        img = read_image(image_path)
        img = self.preprocess(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(img)
        
        return F.softmax(output, dim=1)
    
    def impute_batch(self, image_paths):
        """Impute multiple images in batches.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            predictions: Batched model predictions
        """
        all_outputs = []
        
        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i:i + self.batch_size]
            batch_imgs = []
            
            for path in batch_paths:
                img = read_image(path)
                img = self.preprocess(img)
                batch_imgs.append(img)
            
            batch_tensor = torch.stack(batch_imgs).to(self.device)
            
            with torch.no_grad():
                output = self.model(batch_tensor)
            
            all_outputs.append(F.softmax(output, dim=1))
        
        return torch.cat(all_outputs, dim=0)
    
    def impute_masked_tensor(self, tensor, mask):
        """Impute missing values in a tensor using mask.
        
        Args:
            tensor: Input tensor (B, C, H, W)
            mask: Binary mask (B, C, H, W) where 1 = missing
            
        Returns:
            imputed_tensor: Tensor with imputed values
        """
        tensor = tensor.to(self.device)
        mask = mask.to(self.device)
        
        # Simple imputation: replace masked values with mean
        imputed = tensor.clone()
        imputed[mask == 1] = tensor[mask == 0].mean()
        
        return imputed


# Example usage demonstration
if __name__ == "__main__":
    # Initialize imputer
    imputer = GPUImageImputer(device="cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Device: {imputer.device}")
    print(f"Batch size: {imputer.batch_size}")
    
    # Single image imputation
    image_path = "/home/student/Hackathon/imputer/GoldFish.png"
    predictions = imputer.impute_single(image_path)
    print(f"\nSingle image prediction shape: {predictions.shape}")
    print(f"Top-5 probabilities: {predictions[0].topk(5).values.cpu().numpy()}")

    # Get ImageNet labels
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    with urllib.request.urlopen(url) as f:
        labels = json.loads(f.read().decode())
    
    # Get top prediction
    top_prob, top_idx = predictions[0].topk(1)
    predicted_label = labels[top_idx.item()]
    
    print(f"\n=== Single Image Classification ===")
    print(f"Predicted: {predicted_label} ({top_prob.item()*100:.2f}%)")
    
    # Batch imputation
    batch_paths = [image_path] * 10  # Simulate 10 images
    batch_preds = imputer.impute_batch(batch_paths)
    print(f"\nBatch predictions shape: {batch_preds.shape}")
    
    # Masked tensor imputation
    dummy_tensor = torch.randn(8, 3, 224, 224)
    dummy_mask = torch.randint(0, 2, (8, 3, 224, 224))
    imputed = imputer.impute_masked_tensor(dummy_tensor, dummy_mask)
    print(f"\nImputed tensor shape: {imputed.shape}")
    
    # Test GPU Baseline Imputer
    print("\n=== GPU Baseline Imputer ===")
    data = torch.stack([read_image(image_path).float() for _ in range(5)])
    for strat in ["mean", "median", "constant"]:
        baseline = GPUBaselineImputer(
            model=imputer.model,
            data=data,
            device=imputer.device,
            strategy=strat,  # type: ignore
            constant_value=128.0
        )
        
        # Test with coalition
        test_img = read_image(image_path).float().unsqueeze(0)
        coalition = torch.tensor([True, False, True])
        imputed = baseline.impute_with_coalition(test_img, coalition)
        preds = baseline.predict_batch(imputed)
        
        print(f"{strat.upper()}: prediction shape {preds.shape}, top prob {preds[0].max():.4f}")

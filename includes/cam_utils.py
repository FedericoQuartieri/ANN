"""
Grad-CAM (Gradient-weighted Class Activation Mapping) utilities.

Provides tools for visualizing which regions of input images are most important
for model predictions using gradient-based class activation mapping.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch


class GradCAM:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping) implementation.
    
    Grad-CAM uses gradients flowing into the final convolutional layer
    to produce a coarse localization map highlighting important regions.
    """
    
    def __init__(self, model, target_layer):
        """
        Initialize Grad-CAM.
        
        Args:
            model: Trained classifier
            target_layer: Target convolutional layer for CAM generation
        """
        self.model = model
        self.gradients = None
        self.activations = None
        self.handles = []
        
        # Register forward and backward hooks
        self.handles.append(
            target_layer.register_forward_hook(self.save_activation)
        )
        self.handles.append(
            target_layer.register_full_backward_hook(self.save_gradient)
        )
    
    def save_activation(self, module, input, output):
        """Save activations from forward pass."""
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        """Save gradients from backward pass."""
        self.gradients = grad_output[0]
    
    def cleanup(self):
        """Remove registered hooks."""
        for h in self.handles:
            h.remove()
    
    def __call__(self, x, class_idx=None):
        """
        Generate Grad-CAM heatmap.
        
        Args:
            x: Input tensor (single image)
            class_idx: Target class index (if None, uses predicted class)
        
        Returns:
            Tuple of (normalized_cam, class_index)
        """
        self.model.eval()
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(x)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Backward pass for target class
        output[0, class_idx].backward()
        
        # Compute importance weights (average gradients)
        grads = self.gradients.cpu().numpy()[0]
        acts = self.activations.detach().cpu().numpy()[0]
        weights = np.mean(grads, axis=(1, 2))
        
        # Generate weighted CAM
        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * acts[i]
        
        # Apply ReLU and resize
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (x.shape[3], x.shape[2]))
        
        # Normalize
        return cam / (cam.max() + 1e-8), class_idx


def visualize_gradcam_for_validation(model, val_loader, idx_to_label, device, 
                                      num_samples=10, save_dir=None, exp_name="experiment"):
    """
    Visualize Grad-CAM on validation samples with side-by-side comparison.
    
    Shows original images alongside heatmap overlays for direct comparison.
    
    Args:
        model: Trained model
        val_loader: Validation data loader
        idx_to_label: Dictionary mapping indices to label names
        device: Device (cpu/cuda)
        num_samples: Number of samples to visualize
        save_dir: Optional directory to save visualizations
        exp_name: Experiment name for saving files
    """
    # Get target layer (last conv layer for ResNet50)
    if hasattr(model, 'layer4'):
        target_layer = model.layer4[-1]
    else:
        # For models wrapped in nn.Module
        target_layer = model.layer4[-1] if hasattr(model, 'layer4') else None
        
    if target_layer is None:
        print("Could not find target layer. Model structure:")
        print(model)
        return
    
    # Initialize Grad-CAM
    gcam = GradCAM(model, target_layer)
    
    # Get a batch from validation
    val_iter = iter(val_loader)
    images, labels = next(val_iter)
    
    # ImageNet normalization stats for denormalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    num_samples = min(num_samples, len(images))
    
    # Create figure with 2 columns: original + heatmap overlay
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    plt.suptitle("Grad-CAM: Model Attention Visualization on Validation Set", 
                fontsize=16, fontweight='bold', y=0.995)
    
    for i in range(num_samples):
        # Prepare single image tensor with gradients enabled
        img_t = images[i:i+1].to(device)
        img_t.requires_grad = True
        
        # Generate Grad-CAM heatmap
        mask, pred_idx = gcam(img_t)
        
        # Denormalize image for visualization
        img_np = images[i].cpu().permute(1, 2, 0).numpy()
        img_np = img_np * std + mean
        img_np = np.clip(img_np, 0, 1)
        
        # Convert to uint8
        img_display = (img_np * 255).astype(np.uint8)
        
        # Get labels
        true_label = idx_to_label[labels[i].item()]
        pred_label = idx_to_label[pred_idx]
        
        # Determine if prediction is correct
        is_correct = (labels[i].item() == pred_idx)
        title_color = 'green' if is_correct else 'red'
        
        # Left column: original image
        axes[i, 0].imshow(img_display)
        axes[i, 0].set_title(f"Original\nTrue: {true_label}", 
                            fontsize=11, fontweight='bold')
        axes[i, 0].axis('off')
        
        # Right column: heatmap overlay
        axes[i, 1].imshow(img_display)
        axes[i, 1].imshow(mask, cmap='jet', alpha=0.5)
        axes[i, 1].set_title(f"Grad-CAM Heatmap\nPred: {pred_label}", 
                            fontsize=11, fontweight='bold', color=title_color)
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"gradcam_{exp_name}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved Grad-CAM visualization to: {save_path}")
    
    plt.show()
    
    # Cleanup
    gcam.cleanup()

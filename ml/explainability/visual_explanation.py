"""Visual Explainability: Grad-CAM heatmap visualization for visual manipulation regions."""

import io
import base64
from typing import Optional, Tuple, Dict, Any
from PIL import Image
import numpy as np
import cv2
import torch
import torch.nn.functional as F


class VisualGradCAM:
    """Computes Grad-CAM attention heatmaps on the final convolutional layer of ResNet18."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        """Hook into the last convolutional block of the vision encoder."""
        # Vision encoder conv_stem[-1] is layer4 of ResNet18
        target_layer = self.model.vision_encoder.conv_stem[-1]

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        h1 = target_layer.register_forward_hook(forward_hook)
        h2 = target_layer.register_full_backward_hook(backward_hook)
        self._hook_handles.extend([h1, h2])

    def generate_cam_overlay(
        self,
        image_tensor: torch.Tensor,
        pil_image: Image.Image,
        target_class: int = 1,
        alpha: float = 0.5,
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """Generate Grad-CAM heatmap and overlay on the original PIL image.

        Args:
            image_tensor: [1, 3, H, W]
            pil_image: Original PIL Image
            target_class: Class to explain (default 1: FAKE)
            alpha: Heatmap blend opacity

        Returns:
            Tuple of (base64_data_uri, metadata_dict)
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass through vision branch
        vis_out = self.model.vision_encoder(image_tensor)
        pooled = vis_out["projected_pooled"]

        if hasattr(self.model, "vision_classifier"):
            logits = self.model.vision_classifier(pooled)
        elif hasattr(self.model, "classifier"):
            logits = self.model.classifier(pooled)
        else:
            return None, {"error": "Classifier head not found for CAM"}

        target_score = logits[0, target_class]
        target_score.backward(retain_graph=False)

        if self.gradients is None or self.activations is None:
            return None, {"error": "Could not extract gradients or activations"}

        # Global average pooling on gradients across spatial dimensions
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # [1, 1, H, W]
        cam = F.relu(cam)  # Apply ReLU to keep only positive influence

        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        # Resize CAM to match original image dimensions
        orig_w, orig_h = pil_image.size
        cam_resized = cv2.resize(cam, (orig_w, orig_h))
        cam_uint8 = np.uint8(255 * cam_resized)

        # Apply JET colormap
        heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        # Blend with original image
        orig_np = np.array(pil_image.convert("RGB"))
        blended = np.uint8(alpha * heatmap_rgb + (1.0 - alpha) * orig_np)

        # Convert blended image to base64 string
        blended_pil = Image.fromarray(blended)
        buffer = io.BytesIO()
        blended_pil.save(buffer, format="PNG")
        b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        data_uri = f"data:image/png;base64,{b64_str}"

        # Calculate high-activation region coordinates
        threshold = 0.7
        high_act_pixels = np.sum(cam_resized > threshold) / (orig_w * orig_h)

        meta = {
            "type": "model-derived",
            "method": "Grad-CAM (layer4)",
            "high_activation_area_fraction": round(float(high_act_pixels), 4),
            "target_class": target_class,
        }

        return data_uri, meta

    def remove_hooks(self):
        """Clean up hook handles."""
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

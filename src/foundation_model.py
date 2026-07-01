import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import GeospatialConfig

# Define local PyTorch-based LoRA layer to bypass external package requirements
class LoraLinear(nn.Module):
    """
    Applies Low-Rank Adaptation (LoRA) to a standard linear layer.
    """
    def __init__(self, linear_layer: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.05):
        super().__init__()
        self.linear = linear_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        # Freeze base weights
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
            
        in_features = linear_layer.in_features
        out_features = linear_layer.out_features
        
        # LoRA adapters
        self.lora_A = nn.Parameter(torch.empty((r, in_features)))
        self.lora_B = nn.Parameter(torch.empty((out_features, r)))
        self.dropout = nn.Dropout(p=dropout)
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.linear(x)
        # Calculate B * A * x
        lora_output = self.dropout(x) @ self.lora_A.t() @ self.lora_B.t() * self.scaling
        return base_output + lora_output


class Conv3DFallbackEncoder(nn.Module):
    """
    Fallback spatial-temporal encoder using 3D Convolutions.
    Used when offline or unable to download Prithvi weights.
    Accepts (B, Channels=6, Timesteps=T, H=224, W=224)
    Outputs spatial-temporal embeddings of shape (B, EmbeddingDim=768, H=224, W=224)
    """
    def __init__(self, in_channels: int = 6, embed_dim: int = 768):
        super().__init__()
        # Conv3D to project channels and compress time
        # Kernel size (3, 3, 3), stride (1, 1, 1), padding (1, 1, 1)
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        
        # Final projection to match embedding dimension
        self.proj = nn.Conv2d(256, embed_dim, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))  # (B, 256, T, H, W)
        
        # Pool temporal dimension
        x = torch.mean(x, dim=2)   # (B, 256, H, W)
        x = self.proj(x)           # (B, 768, H, W)
        return x


class PrithviDownstreamAdapter(nn.Module):
    """
    Downstream adapter head that takes Prithvi or Fallback embeddings 
    and classifies crop types and growth (phenology) stages.
    """
    def __init__(self, embed_dim: int = 768, num_crop_classes: int = 5, num_stages: int = 4):
        super().__init__()
        # Dense spatial classification layers
        self.crop_head_conv = nn.Conv2d(embed_dim, 256, kernel_size=3, padding=1)
        self.crop_classifier = nn.Conv2d(256, num_crop_classes, kernel_size=1)
        
        self.stage_head_conv = nn.Conv2d(embed_dim, 256, kernel_size=3, padding=1)
        self.stage_classifier = nn.Conv2d(256, num_stages, kernel_size=1)
        
        # Wrap classification linear projections in LoRA if they were standard Linear layers, 
        # but here we use Conv2d for pixel-level outputs. Let's add LoRA capability to linear decoders if needed.
        
    def forward(self, embeddings: torch.Tensor) -> tuple:
        """
        Input: embeddings of shape (B, embed_dim, H, W)
        Outputs: 
          - crop_logits: (B, num_crop_classes, H, W)
          - stage_logits: (B, num_stages, H, W)
        """
        # Crop classification track
        c_feat = F.relu(self.crop_head_conv(embeddings))
        crop_logits = self.crop_classifier(c_feat)
        
        # Phenological growth stage track
        s_feat = F.relu(self.stage_head_conv(embeddings))
        stage_logits = self.stage_classifier(s_feat)
        
        return crop_logits, stage_logits


def load_foundation_model(use_fallback: bool = False) -> tuple:
    """
    Loads Prithvi-100M from Hugging Face. If load fails or offline,
    loads the Conv3DFallbackEncoder.
    Returns: (encoder_model, embed_dim, is_fallback)
    """
    model_name = GeospatialConfig.MODEL_NAME
    
    if use_fallback:
        print("[Foundation Model] Forcing Conv3D Fallback Encoder.")
        return Conv3DFallbackEncoder(in_channels=GeospatialConfig.NUM_CHANNELS), 768, True
        
    try:
        from transformers import AutoModel, AutoConfig
        print(f"[Foundation Model] Attempting to load {model_name} from Hugging Face...")
        
        # Load configuration and model
        # Prithvi uses custom remote architecture code (ViT Masked Autoencoder)
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True, config=config)
        
        embed_dim = getattr(config, "hidden_size", 768)
        print(f"[Foundation Model] Successfully loaded Prithvi model. Embedding Dim: {embed_dim}")
        
        # Apply LoRA to model's attention linear projections to freeze base and make it PEFT-ready
        lora_applied_count = 0
        for name, module in model.named_modules():
            # Apply to QKV projections in attention layers
            if "qkv" in name.lower() and isinstance(module, nn.Linear):
                parent_name = ".".join(name.split(".")[:-1])
                target_attr = name.split(".")[-1]
                parent_module = dict(model.named_modules())[parent_name]
                
                # Replace with LoraLinear
                lora_layer = LoraLinear(module, r=GeospatialConfig.LORA_R, alpha=GeospatialConfig.LORA_ALPHA)
                setattr(parent_module, target_attr, lora_layer)
                lora_applied_count += 1
                
        print(f"[Foundation Model] LoRA adapter applied to {lora_applied_count} layers.")
        return model, embed_dim, False
        
    except Exception as e:
        print(f"[Foundation Model] Hugging Face load failed: {e}")
        print("[Foundation Model] Switching to Conv3DFallbackEncoder.")
        return Conv3DFallbackEncoder(in_channels=GeospatialConfig.NUM_CHANNELS), 768, True


def train_adapter_epoch(
    encoder: nn.Module, 
    adapter: nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    optimizer: torch.optim.Optimizer, 
    device: torch.device,
    use_fp16: bool = True
) -> float:
    """
    Trains the downstream adapter head (and LoRA weights) for one epoch.
    Implements Mixed Precision (FP16) training for efficiency.
    """
    encoder.train()
    adapter.train()
    
    # Enable gradient scaler for FP16 mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    
    total_loss = 0.0
    for batch_idx, (inputs, crop_labels, stage_labels) in enumerate(dataloader):
        # inputs: (B, 6, T, H, W)
        # crop_labels: (B, H, W)
        # stage_labels: (B, H, W)
        inputs = inputs.to(device)
        crop_labels = crop_labels.to(device).long()
        stage_labels = stage_labels.to(device).long()
        
        optimizer.zero_grad()
        
        # Autocast FP16 contexts
        with torch.cuda.amp.autocast(enabled=use_fp16):
            # Extract embeddings
            # In Prithvi model, the forward expects shape (B, C, T, H, W) or (B, T, C, H, W)
            # Depending on configuration, if fallback: returns (B, 768, H, W) directly
            # If standard Prithvi: we reshape to patch embeddings, then project to spatial map
            embeddings = encoder(inputs)
            
            # If standard Prithvi returns patch tokens (B, L, D), reshape back to spatial layout (B, D, H_p, W_p)
            if len(embeddings.shape) == 3:
                # Shape is (B, L, D). Assuming HLS model output spatial patching
                b_size, seq_len, d_dim = embeddings.shape
                # Prithvi-100m patch size is 16x16, temporal dimension is pooled or kept
                # Number of spatial patches = (224/16) * (224/16) = 14 * 14 = 196
                # If sequence has temporal length, seq_len = T * 196
                spatial_patches = 14
                if seq_len % (spatial_patches * spatial_patches) == 0:
                    t_patches = seq_len // (spatial_patches * spatial_patches)
                    # Reshape and average pool temporal dimension
                    embeddings = embeddings.view(b_size, t_patches, spatial_patches, spatial_patches, d_dim)
                    embeddings = torch.mean(embeddings, dim=1) # (B, 14, 14, 768)
                    embeddings = embeddings.permute(0, 3, 1, 2) # (B, 768, 14, 14)
                    
            # Upsample embeddings to image size if they are downsampled spatial patches (e.g. 14x14 -> 224x224)
            if embeddings.shape[-1] != GeospatialConfig.TILE_SIZE:
                embeddings = F.interpolate(embeddings, size=(GeospatialConfig.TILE_SIZE, GeospatialConfig.TILE_SIZE), mode='bilinear', align_corners=False)
                
            crop_logits, stage_logits = adapter(embeddings)
            
            # Calculate dual loss
            loss_crop = F.cross_entropy(crop_logits, crop_labels, ignore_index=-1)
            loss_stage = F.cross_entropy(stage_logits, stage_labels, ignore_index=-1)
            loss = loss_crop + loss_stage
            
        # Backward pass with FP16 scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

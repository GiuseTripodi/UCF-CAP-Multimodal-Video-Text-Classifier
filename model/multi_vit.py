import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from video_transformer import SpaceTimeTransformer


class TextEncoder(nn.Module):
    """Extract text embeddings using a pre-trained transformer"""

    def __init__(self, model_name='distilbert-base-uncased', embed_dim=768):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.embed_dim = embed_dim

        # Project to match video embedding dimension if needed
        model_dim = self.model.config.hidden_size
        if model_dim != embed_dim:
            self.proj = nn.Linear(model_dim, embed_dim)
        else:
            self.proj = nn.Identity()

    def forward(self, text_list):
        """
        Args:
            text_list: list of strings
        Returns:
            embeddings: (batch_size, embed_dim)
        """
        encoded = self.tokenizer(
            text_list,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )

        with torch.no_grad():
            output = self.model(**encoded)

        # Use [CLS] token as sentence representation
        cls_embeddings = output.last_hidden_state[:, 0, :]
        return self.proj(cls_embeddings)


class MultimodalSpaceTimeTransformer(nn.Module):
    """Combines video and text for multimodal classification"""

    def __init__(self,
                 img_size=96,
                 patch_size=16,
                 in_chans=1,
                 num_classes=4,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 num_frames=8,
                 text_model='distilbert-base-uncased',
                 fusion_method='concat'):
        super().__init__()

        self.embed_dim = embed_dim
        self.fusion_method = fusion_method

        # Video encoder (simplified SpaceTimeTransformer)
        self.video_encoder = SpaceTimeTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=0,  # No classification head yet
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            num_frames=num_frames
        )

        # Text encoder
        self.text_encoder = TextEncoder(model_name=text_model, embed_dim=embed_dim)

        # Fusion layer
        if fusion_method == 'concat':
            fusion_dim = embed_dim * 2
        elif fusion_method == 'add':
            fusion_dim = embed_dim
        elif fusion_method == 'cross_attention':
            fusion_dim = embed_dim
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

        # MLP head for classification
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, video_input, text_input):
        """
        Args:
            video_input: (batch_size, num_frames, channels, height, width)
            text_input: list of strings
        Returns:
            logits: (batch_size, num_classes)
        """
        # Extract embeddings
        video_emb = self.video_encoder.forward_features(video_input)  # (batch_size, embed_dim)
        text_emb = self.text_encoder(text_input)  # (batch_size, embed_dim)

        # Fuse modalities
        if self.fusion_method == 'concat':
            fused = torch.cat([video_emb, text_emb], dim=1)
        elif self.fusion_method == 'add':
            fused = video_emb + text_emb
        elif self.fusion_method == 'cross_attention':
            # Simple cross-attention: video attends to text
            attn_weights = torch.softmax(torch.bmm(
                video_emb.unsqueeze(1),
                text_emb.unsqueeze(2)
            ), dim=-1)
            fused = video_emb + attn_weights.squeeze() * text_emb

        # Classification
        logits = self.classifier(fused)
        return logits


# ============ USAGE EXAMPLE ============
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize model
    model = MultimodalSpaceTimeTransformer(
        img_size=96,
        patch_size=16,
        in_chans=1,
        num_classes=4,
        embed_dim=768,
        depth=12,
        num_heads=12,
        num_frames=8,
        text_model='distilbert-base-uncased',
        fusion_method='concat'  # or 'add', 'cross_attention'
    ).to(device)

    # Dummy input
    batch_size = 2
    video = torch.randn(batch_size, 8, 1, 96, 96).to(device)
    text = [
        "CT scan showing nodule in right upper lobe",
        "Normal chest CT examination"
    ]

    # Forward pass
    logits = model(video, text)
    print(f"Output shape: {logits.shape}")  # (2, 4)

    # Loss and optimization
    labels = torch.tensor([0, 1]).to(device)
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits, labels)
    print(f"Loss: {loss.item()}")
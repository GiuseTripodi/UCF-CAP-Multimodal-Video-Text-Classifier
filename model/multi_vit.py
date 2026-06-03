import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig


class TextEncoder(nn.Module):
    """Extract text embeddings using a pre-trained transformer"""

    def __init__(self, model_name='distilbert-base-uncased', embed_dim=768, trainable=False):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.embed_dim = embed_dim
        self.trainable = trainable

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

        device = next(self.model.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}

        if self.trainable:
            output = self.model(**encoded)
        else:
            with torch.no_grad():
                output = self.model(**encoded)

        # Use [CLS] token as sentence representation
        cls_embeddings = output.last_hidden_state[:, 0, :]
        return self.proj(cls_embeddings)


def _load_video_encoder(model_name: str) -> nn.Module:
    config = AutoConfig.from_pretrained(model_name)
    config.output_hidden_states = True
    return AutoModel.from_pretrained(model_name, config=config)


def _freeze_video_backbone(model: nn.Module, trainable_layers: int) -> None:
    for param in model.parameters():
        param.requires_grad = False

    if trainable_layers <= 0:
        return

    encoder = getattr(model, 'encoder', None)
    if encoder is None or not hasattr(encoder, 'layer'):
        return

    for layer in encoder.layer[-trainable_layers:]:
        for param in layer.parameters():
            param.requires_grad = True


class MultimodalSpaceTimeTransformer(nn.Module):
    """Combines video and text for multimodal classification"""

    def __init__(self,
                 num_classes=4,
                 video_model_name='facebook/timesformer-base-finetuned-k400',
                 text_model='distilbert-base-uncased',
                 fusion_method='concat',
                 freeze_video_backbone=True,
                 trainable_layers=2):
        super().__init__()

        self.video_encoder = _load_video_encoder(video_model_name)
        self.embed_dim = self.video_encoder.config.hidden_size
        self.fusion_method = fusion_method

        if freeze_video_backbone:
            _freeze_video_backbone(self.video_encoder, trainable_layers)

        # Text encoder
        self.text_encoder = TextEncoder(model_name=text_model, embed_dim=self.embed_dim)

        # Fusion layer
        if fusion_method == 'concat':
            fusion_dim = self.embed_dim * 2
        elif fusion_method in {'add', 'cross_attention'}:
            fusion_dim = self.embed_dim
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

        # MLP head for classification
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, self.embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.embed_dim, num_classes)
        )

    def forward(self, video_input, text_input):
        """
        Args:
            video_input: (batch_size, num_frames, channels, height, width)
            text_input: list of strings
        Returns:
            logits: (batch_size, num_classes)
        """
        outputs = self.video_encoder(video_input, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1] if outputs.hidden_states else outputs.last_hidden_state
        video_emb = last_hidden[:, 0, :]  # CLS token
        text_emb = self.text_encoder(text_input)  # (batch_size, embed_dim)

        # Fuse modalities
        if self.fusion_method == 'concat':
            fused = torch.cat([video_emb, text_emb], dim=1)
        elif self.fusion_method == 'add':
            fused = video_emb + text_emb
        elif self.fusion_method == 'cross_attention':
            attn_weights = torch.softmax(torch.bmm(
                video_emb.unsqueeze(1),
                text_emb.unsqueeze(2)
            ), dim=-1)
            fused = video_emb + attn_weights.squeeze() * text_emb

        logits = self.classifier(fused)
        return logits

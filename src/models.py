"""
Kiến trúc mô hình học sâu cho bài toán Phát hiện tin giả tiếng Việt:
1. FocalLoss: Xử lý mất cân bằng dữ liệu (imbalanced classification).
2. PhoBERTWithMetadata: Mô hình kết hợp văn bản (PhoBERT CLS embedding)
   và đặc trưng số mạng xã hội (Metadata MLP) để tăng độ chính xác phân loại.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, PreTrainedModel, AutoConfig

class FocalLoss(nn.Module):
    """
    Focal Loss: Giảm trọng số các mẫu dễ phân loại, tập trung vào các mẫu khó.
    Công thức: FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss

class PhoBERTWithMetadata(nn.Module):
    """
    Kiến trúc kết hợp Đa đặc trưng (Text + Social Metadata):
    - Text Backbone: PhoBERT (vinai/phobert-base-v2) trích xuất vector [CLS] đại diện toàn câu (dim: 768).
    - Metadata Branch: Mạng nơ-ron nhiều tầng (MLP) học đặc trưng số (likes, shares, comments, is_verified).
    - Fusion Head: Nối (Concatenate) 2 biểu diễn lại để phân loại nhị phân.
    """
    def __init__(
        self,
        model_name: str = "vinai/phobert-base-v2",
        num_meta_features: int = 4,
        num_labels: int = 2,
        dropout_rate: float = 0.2,
        use_focal_loss: bool = False
    ):
        super(PhoBERTWithMetadata, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.roberta = AutoModel.from_pretrained(model_name, config=self.config)
        
        # Nhánh xử lý đặc trưng metadata số
        hidden_dim = self.config.hidden_size  # 768 với PhoBERT-base
        self.meta_mlp = nn.Sequential(
            nn.Linear(num_meta_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Tầng phân loại kết hợp (768 + 32 = 800)
        fusion_dim = hidden_dim + 32
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_labels)
        )
        
        self.use_focal_loss = use_focal_loss
        self.loss_fn = FocalLoss() if use_focal_loss else nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        metadata: torch.Tensor = None,
        labels: torch.Tensor = None
    ):
        # 1. Trích xuất đặc trưng văn bản từ Transformer
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        # Lấy vector đại diện [CLS] (token đầu tiên ở index 0)
        cls_rep = outputs.last_hidden_state[:, 0, :]  # Shape: (batch_size, 768)

        # 2. Xử lý metadata số
        if metadata is not None:
            meta_rep = self.meta_mlp(metadata)         # Shape: (batch_size, 32)
            # Nối vector biểu diễn ngữ nghĩa và vector tương tác mạng xã hội
            combined_rep = torch.cat([cls_rep, meta_rep], dim=1) # Shape: (batch_size, 800)
        else:
            # Nếu không có metadata, đệm 0 cho phần meta
            zero_meta = torch.zeros((cls_rep.size(0), 32), device=cls_rep.device)
            combined_rep = torch.cat([cls_rep, zero_meta], dim=1)

        # 3. Qua tầng phân loại
        logits = self.classifier(combined_rep)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}


import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model


# ==========================================
# 1. THE TEACHER: Wav2Vec 2.0 (Massive, SOTA)
# ==========================================
class Wav2VecTeacher(nn.Module):
    def __init__(self):
        super(Wav2VecTeacher, self).__init__()
        print("Loading Wav2Vec 2.0 Base Model from Hugging Face...")
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

        # Freeze the entire Wav2Vec backbone
        for param in self.wav2vec.parameters():
            param.requires_grad = False

        # LayerNorm stabilizes wav2vec hidden states before classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        # x shape: [Batch, 64000]
        outputs      = self.wav2vec(x)
        hidden_states = outputs.last_hidden_state.mean(dim=1)
        logits        = self.classifier(hidden_states)
        return logits


# ==========================================
# HELPER BLOCKS FOR STUDENT
# ==========================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation: learns which channels matter most."""
    def __init__(self, channels, reduction=8):
        super(SEBlock, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.se(x).unsqueeze(-1).unsqueeze(-1)
        return x * scale


class ResBlock(nn.Module):
    """Residual block with SE attention — prevents vanishing gradients."""
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.se   = SEBlock(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out      = self.block(x)
        out      = self.se(out)
        out      = out + residual
        out      = self.relu(out)
        return out


# ==========================================
# 2. THE STUDENT: Mobile CNN (Tiny, Fast)
# ==========================================
class MobileStudentCNN(nn.Module):
    def __init__(self):
        super(MobileStudentCNN, self).__init__()

        # Block 1 — Stem
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Block 2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Block 3
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3   = nn.BatchNorm2d(64)

        # Residual + SE attention block
        self.res_block     = ResBlock(64)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Classifier head
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        # x shape: [Batch, 1, 64, 126]
        x      = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x      = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x      = F.relu(self.bn3(self.conv3(x)))
        x      = self.res_block(x)
        x      = self.adaptive_pool(x)
        logits = self.fc(x)
        return logits


# ==========================================
# Quick self-test
# ==========================================
if __name__ == "__main__":
    print("Testing Neural Network Architectures...")

    dummy_audio       = torch.randn(2, 64000)
    dummy_spectrogram = torch.randn(2, 1, 64, 126)

    print("\nInitializing Student CNN...")
    student     = MobileStudentCNN()
    student_out = student(dummy_spectrogram)
    total_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"  Output shape : {student_out.shape}  (Expected: [2, 2])")
    print(f"  Parameters   : {total_params:,}  (~{total_params/1e6:.2f}M)")

    print("\nInitializing Teacher Wav2Vec 2.0...")
    teacher     = Wav2VecTeacher()
    teacher_out = teacher(dummy_audio)
    frozen      = sum(p.numel() for p in teacher.wav2vec.parameters())
    trainable   = sum(p.numel() for p in teacher.classifier.parameters())
    print(f"  Output shape     : {teacher_out.shape}  (Expected: [2, 2])")
    print(f"  Frozen backbone  : {frozen:,}  (~{frozen/1e6:.1f}M)")
    print(f"  Trainable head   : {trainable:,}")

    assert student_out.shape == torch.Size([2, 2])
    assert teacher_out.shape == torch.Size([2, 2])
    print("\n✅ All assertions passed")

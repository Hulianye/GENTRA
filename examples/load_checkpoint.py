"""Load the released checkpoint and run a synthetic-input smoke check."""
from __future__ import annotations
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gentra.models.loader import GENTRAModel
from gentra.models.masked_reconstruction import build_reconstruction_head
from gentra.pipelines.pretrain_runner import make_dummy_eeg, model_kwargs_from_config
from gentra.training.channel_encoding import vocab_size_from_vocab

checkpoint = torch.load(ROOT / "checkpoints/gentra_pretrained.pt", map_location="cpu", weights_only=True)
cfg, vocab = checkpoint["cfg"], checkpoint["vocab"]
model = GENTRAModel(make_dummy_eeg(vocab, n_classes=len(cfg["train_classes"])), channel_vocab=vocab, **model_kwargs_from_config(cfg))
if any(name.startswith("recon_head.") for name in checkpoint["model_state"]):
    model.recon_head = build_reconstruction_head(cfg["recon_target"], emb_size=int(cfg["emb_size"]), conv_dim=int(cfg["conv_filters"]), channel_vocab_size=vocab_size_from_vocab(vocab), recon_hidden_dim=int(cfg.get("recon_hidden_dim", 128)), samples_per_token=int(cfg.get("recon_samples_per_token", 16)))
model.load_state_dict(checkpoint["model_state"])
model.eval()
channels = ["C3", "C4", "CZ", "FZ"]
channel_ids = torch.tensor([[vocab[name] for name in channels]], dtype=torch.long)
channel_mask = torch.ones_like(channel_ids, dtype=torch.bool)
with torch.no_grad():
    logits = model(torch.zeros(1, len(channels), 1000), channel_ids, channel_mask)
print(f"Loaded GENTRA checkpoint; logits shape: {tuple(logits.shape)}")

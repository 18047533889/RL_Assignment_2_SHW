"""
export_model.py — Extract trained weights from a TorchRL checkpoint to flat SuperTTTNet .pt.

No Ray/RLlib dependency. Checkpoints are plain torch state_dicts.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import torch

from network import SuperTTTNet, get_device
from config import MODEL_CONFIG


def export_checkpoint(
    checkpoint_path: str,
    output_path: str,
    *,
    map_location: str | torch.device | None = None,
) -> str:
    checkpoint_path = os.path.abspath(os.path.expanduser(checkpoint_path))
    sd = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    if "main_net" in sd:
        sd = sd["main_net"]

    net = SuperTTTNet(
        in_channels=7,
        num_filters=int(MODEL_CONFIG.get("num_filters", 192)),
        num_res_blocks=int(MODEL_CONFIG.get("num_res_blocks", 8)),
        num_actions=96,
        value_fc_hidden=int(MODEL_CONFIG.get("value_fc_hidden", 768)),
    )
    if map_location is None:
        map_location = get_device()
    missing, _unexpected = net.load_state_dict(sd, strict=False)
    if missing and len(missing) > len(sd) * 0.5:
        raise RuntimeError(
            f"Too many missing keys: missing={missing[:8]}..."
        )
    net.to(map_location)
    out = os.path.abspath(os.path.expanduser(output_path))
    torch.save(net.state_dict(), out)
    return out


def main():
    parser = argparse.ArgumentParser(description="Export trained weights to SuperTTTNet .pt")
    parser.add_argument("checkpoint", help="Checkpoint .pt file")
    parser.add_argument(
        "-o", "--output",
        default="model_weights.pt",
        help="Output path for state_dict (default: model_weights.pt)",
    )
    args = parser.parse_args()
    out = export_checkpoint(args.checkpoint, args.output)
    print(f"Saved SuperTTTNet weights to {out}")


if __name__ == "__main__":
    main()

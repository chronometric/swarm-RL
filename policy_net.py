"""Load SwarmDepthCNN from my_agent/drone_agent.py for SB3-compatible pickling."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def load_swarm_depth_cnn_class():
    """Import SwarmDepthCNN as module ``drone_agent`` (matches Docker submission layout)."""
    agent_path = Path(__file__).resolve().parent.parent / "my_agent" / "drone_agent.py"
    spec = importlib.util.spec_from_file_location("drone_agent", agent_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load drone_agent from {agent_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SwarmDepthCNN

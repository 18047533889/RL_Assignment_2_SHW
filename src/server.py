"""
server.py — Flask API for human vs exported policy (no RLlib in the loop).

**Quick read:** Load ``SuperTTTNet`` weights from ``export_model.py``; build 7-plane
ego obs like ``env.py``; masked logits for AI. Supports selectable first/second player,
three difficulty levels (easy/medium/hard), and model version selection.

Observations match training: 7-channel ego-centric (my pieces, opponent pieces,
valid mask, opponent last move, my threats, opponent threats, forfeit probability).
Actions use the compact 96-dim space via ``board.compact_to_rc`` / ``rc_to_compact``.

Also supports loading old-architecture models (3ch/144-action) for version comparison.

Endpoints:
    POST /reset          → Start a new game (accepts ``ai_first``, ``difficulty``)
    POST /move           → Human makes a move, AI responds
    POST /demo_move      → AI vs AI: both players move (one round)
    GET  /state          → Get current board state
    GET  /api/status     → Whether trained weights are loaded + device / policy mode
    GET  /api/versions   → List available model versions from registry
    POST /api/load_version → Switch to a different model version
    GET  /               → Serve the game frontend
    GET  /report         → Serve ``report.html`` from assignment root
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
from flask import Flask, request, jsonify, send_from_directory

from board import (
    ROWS, COLS, VALID_MASK, VALID_POSITIONS, NUM_VALID,
    POS_TO_INDEX, compact_to_rc, rc_to_compact, empty_board,
)
from rules import check_win, is_draw, threat_heatmap
from stochastic import resolve_placement, forfeit_probability_map
from env import OBS_CHANNELS, heuristic_action, random_legal_action
from eval_rollout import lookahead_action
from network import SuperTTTNet, get_device
from config import MODEL_CONFIG

ROOT_DIR = os.path.join(os.path.dirname(__file__), "..")
app = Flask(__name__, static_folder=os.path.join(ROOT_DIR, "frontend"))

game_state = {
    "board": empty_board(),
    "current_player": 1,
    "game_over": False,
    "winner": None,
    "rng": np.random.default_rng(42),
    "history": [],
    "human_player": 1,
    "ai_player": 2,
    "difficulty": "hard",
    "last_move_plane": np.zeros((ROWS, COLS), dtype=np.float32),
}

agent_net = None
agent_device: torch.device | None = None
loaded_weights_path: str | None = None
loaded_arch: str = "new"
ai_stochastic: bool = False
MODELS_DIR = os.path.join(ROOT_DIR, "models")


def _net_kwargs() -> dict:
    return {
        "in_channels": OBS_CHANNELS,
        "num_filters": int(MODEL_CONFIG.get("num_filters", 192)),
        "num_res_blocks": int(MODEL_CONFIG.get("num_res_blocks", 8)),
        "num_actions": NUM_VALID,
        "value_fc_hidden": int(MODEL_CONFIG.get("value_fc_hidden", 768)),
    }


def _detect_arch(path: str) -> str:
    version_dir = os.path.dirname(path)
    info_path = os.path.join(version_dir, "version_info.json")
    if os.path.isfile(info_path):
        with open(info_path, encoding="utf-8") as f:
            info = json.load(f)
        arch = info.get("architecture", {})
        if isinstance(arch, dict) and arch.get("obs_channels", 7) == 3:
            return "old"
        if arch == "old":
            return "old"
    return "new"


def load_agent(checkpoint_path: str | None = None) -> None:
    global agent_net, agent_device, loaded_weights_path, loaded_arch
    agent_device = get_device()
    loaded_weights_path = None
    agent_net = None
    loaded_arch = "new"

    path = (checkpoint_path or "").strip()
    if path and not os.path.isfile(path):
        print(f"Warning: model file not found ({path}). AI will use random legal moves.")

    if path and os.path.isfile(path):
        loaded_arch = _detect_arch(path)
        if loaded_arch == "old":
            agent_net = SuperTTTNet(
                in_channels=3, num_filters=128, num_res_blocks=6,
                num_actions=144, value_fc_hidden=512,
            )
        else:
            agent_net = SuperTTTNet(**_net_kwargs())
        state_dict = torch.load(path, map_location=agent_device, weights_only=True)
        agent_net.load_state_dict(state_dict, strict=False)
        agent_net.to(agent_device)
        agent_net.eval()
        loaded_weights_path = os.path.abspath(path)
        print(f"Loaded weights: {loaded_weights_path} | arch={loaded_arch} | device={agent_device}")
    else:
        print("No valid --model: AI uses uniform random legal moves (not an untrained network).")


def build_obs(board: np.ndarray, player: int, last_move_plane: np.ndarray) -> np.ndarray:
    pid = player
    opp = 3 - pid
    obs = np.zeros((OBS_CHANNELS, ROWS, COLS), dtype=np.float32)
    obs[0] = (board == pid).astype(np.float32)
    obs[1] = (board == opp).astype(np.float32)
    obs[2] = VALID_MASK.astype(np.float32)
    obs[3] = last_move_plane
    my_hmap = threat_heatmap(board, pid)
    opp_hmap = threat_heatmap(board, opp)
    max_val = max(my_hmap.max(), opp_hmap.max(), 1.0)
    obs[4] = my_hmap / max_val
    obs[5] = opp_hmap / max_val
    obs[6] = forfeit_probability_map(board)
    return obs


def get_action_mask(board: np.ndarray) -> np.ndarray:
    mask = np.zeros(NUM_VALID, dtype=np.float32)
    for idx, (r, c) in enumerate(VALID_POSITIONS):
        if board[r, c] == 0:
            mask[idx] = 1.0
    return mask


def _build_obs_old(board: np.ndarray, player: int) -> np.ndarray:
    pid = player
    opp = 3 - pid
    obs = np.zeros((3, ROWS, COLS), dtype=np.float32)
    obs[0] = (board == pid).astype(np.float32)
    obs[1] = (board == opp).astype(np.float32)
    valid = np.zeros((ROWS, COLS), dtype=np.float32)
    for r in range(ROWS):
        for c in range(COLS):
            if VALID_MASK[r, c] and board[r, c] == 0:
                valid[r, c] = 1.0
    obs[2] = valid
    return obs


def _get_action_mask_flat(board: np.ndarray) -> np.ndarray:
    mask = np.zeros(ROWS * COLS, dtype=np.float32)
    for r in range(ROWS):
        for c in range(COLS):
            if VALID_MASK[r, c] and board[r, c] == 0:
                mask[r * COLS + c] = 1.0
    return mask


def agent_choose_action(board: np.ndarray, player: int,
                        last_move_plane: np.ndarray,
                        difficulty: str = "hard") -> int:
    if difficulty == "easy":
        return random_legal_action(board, game_state["rng"])
    if difficulty == "medium":
        return heuristic_action(board, player, game_state["rng"])

    if agent_net is None:
        return heuristic_action(board, player, game_state["rng"])

    if loaded_arch == "old":
        obs = _build_obs_old(board, player)
        mask = _get_action_mask_flat(board)
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(agent_device)
            logits, _ = agent_net(obs_t)
            mask_t = torch.from_numpy(mask).unsqueeze(0).to(agent_device)
            inf_mask = torch.clamp(torch.log(mask_t + 1e-10), min=-1e10)
            masked = logits + inf_mask
            action_idx = int(torch.argmax(masked, dim=-1).item())
        r, c = divmod(action_idx, COLS)
        if (r, c) in POS_TO_INDEX:
            return POS_TO_INDEX[(r, c)]
        return random_legal_action(board, game_state["rng"])

    obs = build_obs(board, player, last_move_plane)
    mask = get_action_mask(board)
    obs_dict = {"observations": obs, "action_mask": mask}

    if not ai_stochastic:
        return lookahead_action(agent_net, obs_dict, board, player, agent_device)

    with torch.no_grad():
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(agent_device)
        logits, _ = agent_net(obs_t)
        mask_t = torch.from_numpy(mask).unsqueeze(0).to(agent_device)
        inf_mask = torch.clamp(torch.log(mask_t + 1e-10), min=-1e10)
        masked = logits + inf_mask
        probs = torch.softmax(masked, dim=-1).cpu()
        action_idx = torch.multinomial(probs, 1).item()
    return action_idx


def make_move(board: np.ndarray, compact_action: int, player: int,
              rng: np.random.Generator) -> dict:
    r, c = compact_to_rc(compact_action)
    result = {"chosen": (r, c), "placed": None, "forfeited": False}

    if not VALID_MASK[r, c] or board[r, c] != 0:
        result["forfeited"] = True
        return result

    placement = resolve_placement(board, r, c, rng)
    if placement is None:
        result["forfeited"] = True
    else:
        pr, pc = placement
        board[pr, pc] = player
        result["placed"] = (pr, pc)

    return result


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "game.html")


@app.route("/report")
def report_page():
    return send_from_directory(ROOT_DIR, "report.html")


@app.route("/api/status", methods=["GET"])
def api_status():
    return jsonify(
        {
            "model_loaded": agent_net is not None,
            "model_path": loaded_weights_path,
            "arch": loaded_arch,
            "device": str(agent_device) if agent_device is not None else None,
            "policy": "stochastic" if ai_stochastic else "greedy_argmax",
        }
    )


@app.route("/api/versions", methods=["GET"])
def api_versions():
    reg_path = os.path.join(MODELS_DIR, "registry.json")
    if not os.path.isfile(reg_path):
        return jsonify({"versions": [], "current": loaded_weights_path})
    with open(reg_path, encoding="utf-8") as f:
        reg = json.load(f)
    versions = []
    for v in reg.get("versions", []):
        versions.append({
            "version": v["version"],
            "name": v.get("name", v["version"]),
            "architecture": v.get("architecture", "new"),
            "iterations": v.get("iterations", 0),
        })
    return jsonify({"versions": versions, "current": loaded_weights_path})


@app.route("/api/load_version", methods=["POST"])
def api_load_version():
    data = request.get_json(silent=True) or {}
    version = data.get("version", "")
    if not version:
        return jsonify({"error": "Missing version"}), 400
    weights_path = os.path.join(MODELS_DIR, f"v{version}", "model_weights.pt")
    if not os.path.isfile(weights_path):
        return jsonify({"error": f"No weights for version {version}"}), 404
    load_agent(weights_path)
    return jsonify({
        "status": "ok",
        "version": version,
        "arch": loaded_arch,
        "model_loaded": agent_net is not None,
    })


@app.route("/reset", methods=["POST"])
def reset():
    data = request.get_json(silent=True) or {}
    ai_first = data.get("ai_first", False)
    difficulty = data.get("difficulty", "hard")

    game_state["board"] = empty_board()
    game_state["game_over"] = False
    game_state["winner"] = None
    game_state["rng"] = np.random.default_rng()
    game_state["history"] = []
    game_state["last_move_plane"] = np.zeros((ROWS, COLS), dtype=np.float32)
    game_state["difficulty"] = difficulty

    if ai_first:
        game_state["human_player"] = 2
        game_state["ai_player"] = 1
        game_state["current_player"] = 1

        ai_action = agent_choose_action(
            game_state["board"], 1, game_state["last_move_plane"], difficulty,
        )
        ai_result = make_move(game_state["board"], ai_action, 1, game_state["rng"])
        game_state["history"].append({"player": 1, **ai_result})

        if ai_result["placed"]:
            pr, pc = ai_result["placed"]
            game_state["last_move_plane"] = np.zeros((ROWS, COLS), dtype=np.float32)
            game_state["last_move_plane"][pr, pc] = 1.0
        else:
            game_state["last_move_plane"] = np.zeros((ROWS, COLS), dtype=np.float32)

        game_state["current_player"] = 2

        return jsonify({
            "status": "ok",
            "board": game_state["board"].tolist(),
            "ai_first_move": ai_result,
            "human_player": 2,
            "ai_player": 1,
        })
    else:
        game_state["human_player"] = 1
        game_state["ai_player"] = 2
        game_state["current_player"] = 1

        return jsonify({
            "status": "ok",
            "board": game_state["board"].tolist(),
            "human_player": 1,
            "ai_player": 2,
        })


@app.route("/state", methods=["GET"])
def state():
    return jsonify({
        "board": game_state["board"].tolist(),
        "current_player": game_state["current_player"],
        "game_over": game_state["game_over"],
        "winner": game_state["winner"],
        "human_player": game_state["human_player"],
        "ai_player": game_state["ai_player"],
    })


@app.route("/demo_move", methods=["POST"])
def demo_move():
    if game_state["game_over"]:
        return jsonify({"error": "Game is over"}), 400

    board = game_state["board"]
    rng = game_state["rng"]
    difficulty = game_state.get("difficulty", "hard")
    response = {"p1_move": None, "p2_move": None, "game_over": False, "winner": None}

    for player in [1, 2]:
        action = agent_choose_action(
            board, player, game_state["last_move_plane"], difficulty,
        )
        result = make_move(board, action, player, rng)
        game_state["history"].append({"player": player, **result})
        key = "p1_move" if player == 1 else "p2_move"
        response[key] = result

        if result["placed"]:
            pr, pc = result["placed"]
            game_state["last_move_plane"] = np.zeros((ROWS, COLS), dtype=np.float32)
            game_state["last_move_plane"][pr, pc] = 1.0
        else:
            game_state["last_move_plane"] = np.zeros((ROWS, COLS), dtype=np.float32)

        win = check_win(board, player)
        if win:
            game_state["game_over"] = True
            game_state["winner"] = player
            response["game_over"] = True
            response["winner"] = player
            response["win_info"] = {"type": win[0], "cells": [list(c) for c in win[1]]}
            break

        if is_draw(board):
            game_state["game_over"] = True
            response["game_over"] = True
            response["winner"] = "draw"
            break

    response["board"] = board.tolist()
    return jsonify(response)


@app.route("/move", methods=["POST"])
def move():
    if game_state["game_over"]:
        return jsonify({"error": "Game is over"}), 400

    data = request.get_json(silent=True) or {}
    row, col = data.get("row"), data.get("col")

    if row is None or col is None:
        return jsonify({"error": "Missing row/col"}), 400

    row, col = int(row), int(col)
    if row < 0 or row >= ROWS or col < 0 or col >= COLS or not VALID_MASK[row, col]:
        return jsonify({"error": "Invalid cell"}), 400

    compact_action = rc_to_compact(row, col)
    board = game_state["board"]
    rng = game_state["rng"]
    human_pid = game_state["human_player"]
    ai_pid = game_state["ai_player"]
    difficulty = game_state.get("difficulty", "hard")

    human_result = make_move(board, compact_action, human_pid, rng)
    game_state["history"].append({"player": human_pid, **human_result})

    if human_result["placed"]:
        pr, pc = human_result["placed"]
        game_state["last_move_plane"] = np.zeros((ROWS, COLS), dtype=np.float32)
        game_state["last_move_plane"][pr, pc] = 1.0
    else:
        game_state["last_move_plane"] = np.zeros((ROWS, COLS), dtype=np.float32)

    response = {"human_move": human_result, "ai_move": None,
                "game_over": False, "winner": None}

    win = check_win(board, human_pid)
    if win:
        game_state["game_over"] = True
        game_state["winner"] = human_pid
        response["game_over"] = True
        response["winner"] = human_pid
        response["win_info"] = {"type": win[0],
                                "cells": [list(c) for c in win[1]]}
        response["board"] = board.tolist()
        return jsonify(response)

    if is_draw(board):
        game_state["game_over"] = True
        response["game_over"] = True
        response["winner"] = "draw"
        response["board"] = board.tolist()
        return jsonify(response)

    ai_action = agent_choose_action(
        board, ai_pid, game_state["last_move_plane"], difficulty,
    )
    ai_result = make_move(board, ai_action, ai_pid, rng)
    game_state["history"].append({"player": ai_pid, **ai_result})
    response["ai_move"] = ai_result

    if ai_result["placed"]:
        pr, pc = ai_result["placed"]
        game_state["last_move_plane"] = np.zeros((ROWS, COLS), dtype=np.float32)
        game_state["last_move_plane"][pr, pc] = 1.0
    else:
        game_state["last_move_plane"] = np.zeros((ROWS, COLS), dtype=np.float32)

    win = check_win(board, ai_pid)
    if win:
        game_state["game_over"] = True
        game_state["winner"] = ai_pid
        response["game_over"] = True
        response["winner"] = ai_pid
        response["win_info"] = {"type": win[0],
                                "cells": [list(c) for c in win[1]]}
    elif is_draw(board):
        game_state["game_over"] = True
        response["game_over"] = True
        response["winner"] = "draw"

    response["board"] = board.tolist()
    return jsonify(response)


def main() -> None:
    global ai_stochastic
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Exported SuperTTTNet weights (.pt from export_model.py). Omit for random legal-move AI.",
    )
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample AI actions from the policy (training-like); default is greedy argmax.",
    )
    args = parser.parse_args()
    ai_stochastic = bool(args.stochastic)
    load_agent(args.model)
    print(f"Server running on http://localhost:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()

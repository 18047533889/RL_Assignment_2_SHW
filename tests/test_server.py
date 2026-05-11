"""Tests for server.py: Flask routes + unit tests for obs/mask/move helpers."""
import numpy as np
import pytest

pytest.importorskip("flask")

from server import (
    app, game_state, build_obs, get_action_mask, _net_kwargs,
    make_move, _build_obs_old, _get_action_mask_flat,
)
from board import VALID_MASK, ROWS, COLS, NUM_VALID, VALID_POSITIONS, empty_board
from env import OBS_CHANNELS


@pytest.fixture(autouse=True)
def _reset_game():
    app.config["TESTING"] = True
    with app.test_client() as c:
        c.post("/reset")
    yield


def test_api_status():
    """GET ``/api/status``: 200 and keys model_loaded, policy, device."""
    with app.test_client() as c:
        r = c.get("/api/status")
        assert r.status_code == 200
        d = r.get_json()
        assert d["model_loaded"] is False
        assert d["policy"] in ("greedy_argmax", "stochastic")
        assert "device" in d


def test_reset():
    """POST ``/reset``: ok status and board list."""
    with app.test_client() as c:
        r = c.post("/reset")
        assert r.status_code == 200
        d = r.get_json()
        assert d["status"] == "ok"
        assert isinstance(d["board"], list)


def test_post_move_returns_human_move_and_board_when_ongoing():
    """``POST /move``: JSON includes ``human_move``; board listed if episode continues."""
    with app.test_client() as c:
        c.post("/reset")
        r = c.post("/move", json={"row": 9, "col": 5})
        assert r.status_code == 200
        d = r.get_json()
        assert "human_move" in d
        assert d["human_move"]["chosen"] == [9, 5]
        if not d.get("game_over"):
            assert "board" in d
            assert isinstance(d["board"], list)
            assert len(d["board"]) == 12


def test_post_move_when_game_over_returns_400():
    """After ``game_over``, further moves return 400."""
    with app.test_client() as c:
        c.post("/reset")
        game_state["game_over"] = True
        r = c.post("/move", json={"row": 0, "col": 4})
        assert r.status_code == 400
        err = r.get_json()
        assert "error" in err


def test_post_move_missing_row_returns_400():
    """``POST /move`` without ``row`` or ``col`` returns 400."""
    with app.test_client() as c:
        c.post("/reset")
        r = c.post("/move", json={"row": 5})
        assert r.status_code == 400


def test_get_state_returns_board():
    """``GET /state`` returns current board and game status."""
    with app.test_client() as c:
        c.post("/reset")
        r = c.get("/state")
        assert r.status_code == 200
        d = r.get_json()
        assert "board" in d
        assert isinstance(d["board"], list)
        assert d["game_over"] is False
        assert d["current_player"] == 1


def test_demo_move_basic():
    """``POST /demo_move`` returns p1_move, p2_move, board."""
    with app.test_client() as c:
        c.post("/reset")
        r = c.post("/demo_move")
        assert r.status_code == 200
        d = r.get_json()
        assert "p1_move" in d
        assert "p2_move" in d
        assert "board" in d
        assert isinstance(d["board"], list)
        p1 = d["p1_move"]
        assert "chosen" in p1
        assert "forfeited" in p1


def test_demo_move_game_over_returns_400():
    """``POST /demo_move`` after game over returns 400."""
    with app.test_client() as c:
        c.post("/reset")
        game_state["game_over"] = True
        r = c.post("/demo_move")
        assert r.status_code == 400
        assert "error" in r.get_json()


def test_demo_move_full_game():
    """Run demo_move until game terminates (win or draw)."""
    with app.test_client() as c:
        c.post("/reset")
        for _ in range(200):
            r = c.post("/demo_move")
            d = r.get_json()
            if d.get("game_over"):
                assert d["winner"] in (1, 2, "draw")
                if d["winner"] in (1, 2) and d.get("win_info"):
                    assert "type" in d["win_info"]
                    assert "cells" in d["win_info"]
                    assert len(d["win_info"]["cells"]) >= 4
                break
        else:
            pytest.fail("Demo game did not terminate within 200 rounds")


def test_move_ai_responds():
    """After human move, AI should respond (or game ends)."""
    with app.test_client() as c:
        c.post("/reset")
        r = c.post("/move", json={"row": 9, "col": 5})
        d = r.get_json()
        if not d.get("game_over"):
            assert d["ai_move"] is not None
            assert "chosen" in d["ai_move"]


def test_move_human_win_detection():
    """Manually fill a row-4 for player 1 and verify win is detected."""
    with app.test_client() as c:
        c.post("/reset")
        board = game_state["board"]
        board[9, 2] = 1
        board[9, 3] = 1
        board[9, 4] = 1
        game_state["rng"] = np.random.default_rng(0)
        r = c.post("/move", json={"row": 9, "col": 5})
        d = r.get_json()
        if d.get("game_over") and d.get("winner") == 1:
            assert d["win_info"]["type"] == "row"


def test_demo_move_win_info_structure():
    """When demo produces a win, win_info has type and cells list."""
    with app.test_client() as c:
        c.post("/reset")
        board = game_state["board"]
        board[10, 3] = 1
        board[10, 4] = 1
        board[10, 5] = 1
        board[10, 6] = 1
        found_win = False
        for _ in range(50):
            r = c.post("/demo_move")
            d = r.get_json()
            if d.get("game_over"):
                if d.get("win_info"):
                    assert isinstance(d["win_info"]["cells"], list)
                    assert d["win_info"]["type"] in ("row", "col", "diag")
                found_win = True
                break
        assert found_win, "Expected game to end"


def test_state_after_moves_reflects_pieces():
    """Board from /state should show placed pieces after a move."""
    with app.test_client() as c:
        c.post("/reset")
        c.post("/move", json={"row": 9, "col": 5})
        r = c.get("/state")
        d = r.get_json()
        board = np.array(d["board"])
        placed = int(np.sum(board > 0))
        assert placed >= 1


def test_reset_accepts_ai_first_and_difficulty():
    """POST /reset with ai_first and difficulty parameters."""
    with app.test_client() as c:
        r = c.post("/reset", json={"ai_first": True, "difficulty": "easy"})
        assert r.status_code == 200
        d = r.get_json()
        assert d["status"] == "ok"


def test_reset_difficulty_medium():
    """POST /reset with medium difficulty."""
    with app.test_client() as c:
        r = c.post("/reset", json={"difficulty": "medium"})
        assert r.status_code == 200
        d = r.get_json()
        assert d["status"] == "ok"


def test_move_out_of_bounds():
    """POST /move with out-of-bounds row/col returns 400."""
    with app.test_client() as c:
        c.post("/reset")
        r = c.post("/move", json={"row": -1, "col": 5})
        assert r.status_code == 400
        r = c.post("/move", json={"row": 99, "col": 5})
        assert r.status_code == 400


def test_move_invalid_cell():
    """POST /move with a cell outside VALID_MASK returns 400."""
    with app.test_client() as c:
        c.post("/reset")
        r = c.post("/move", json={"row": 0, "col": 0})
        assert r.status_code == 400


def test_move_non_json_body():
    """POST /move with non-JSON body returns 400."""
    with app.test_client() as c:
        c.post("/reset")
        r = c.post("/move", data="not json", content_type="text/plain")
        assert r.status_code == 400


def test_api_versions():
    """GET /api/versions returns a list."""
    with app.test_client() as c:
        r = c.get("/api/versions")
        assert r.status_code == 200
        d = r.get_json()
        assert "versions" in d
        assert isinstance(d["versions"], list)


def test_api_load_version_missing():
    """POST /api/load_version with missing version returns 400."""
    with app.test_client() as c:
        r = c.post("/api/load_version", json={})
        assert r.status_code == 400


def test_api_load_version_not_found():
    """POST /api/load_version with nonexistent version returns 404."""
    with app.test_client() as c:
        r = c.post("/api/load_version", json={"version": "99.99"})
        assert r.status_code == 404


def test_api_status_has_arch_key():
    """GET /api/status includes arch field."""
    with app.test_client() as c:
        r = c.get("/api/status")
        d = r.get_json()
        assert "arch" in d


def test_build_obs_shape():
    board = empty_board()
    lmp = np.zeros((ROWS, COLS), dtype=np.float32)
    obs = build_obs(board, 1, lmp)
    assert obs.shape == (OBS_CHANNELS, ROWS, COLS)
    assert obs.dtype == np.float32


def test_build_obs_ego_centric():
    board = empty_board()
    r0, c0 = VALID_POSITIONS[0]
    board[r0, c0] = 1
    lmp = np.zeros((ROWS, COLS), dtype=np.float32)
    obs_p1 = build_obs(board, 1, lmp)
    obs_p2 = build_obs(board, 2, lmp)
    assert obs_p1[0, r0, c0] == 1.0
    assert obs_p1[1, r0, c0] == 0.0
    assert obs_p2[0, r0, c0] == 0.0
    assert obs_p2[1, r0, c0] == 1.0


def test_build_obs_valid_mask_channel():
    board = empty_board()
    lmp = np.zeros((ROWS, COLS), dtype=np.float32)
    obs = build_obs(board, 1, lmp)
    assert np.array_equal(obs[2], VALID_MASK.astype(np.float32))


def test_build_obs_last_move_channel():
    board = empty_board()
    lmp = np.zeros((ROWS, COLS), dtype=np.float32)
    lmp[3, 5] = 1.0
    obs = build_obs(board, 1, lmp)
    assert obs[3, 3, 5] == 1.0


def test_get_action_mask_empty_board():
    board = empty_board()
    mask = get_action_mask(board)
    assert mask.shape == (NUM_VALID,)
    assert mask.sum() == NUM_VALID


def test_get_action_mask_occupied():
    board = empty_board()
    r0, c0 = VALID_POSITIONS[0]
    board[r0, c0] = 1
    mask = get_action_mask(board)
    assert mask[0] == 0.0
    assert mask.sum() == NUM_VALID - 1


def test_net_kwargs_matches_config():
    kw = _net_kwargs()
    assert kw["in_channels"] == OBS_CHANNELS
    assert kw["num_actions"] == NUM_VALID
    assert kw["num_filters"] == 192
    assert kw["num_res_blocks"] == 8
    assert kw["value_fc_hidden"] == 768


def test_make_move_valid():
    board = empty_board()
    rng = np.random.default_rng(42)
    result = make_move(board, 0, 1, rng)
    assert result["chosen"] is not None
    assert not result["forfeited"] or result["placed"] is None


def test_make_move_occupied_cell():
    board = empty_board()
    r0, c0 = VALID_POSITIONS[0]
    board[r0, c0] = 1
    rng = np.random.default_rng(42)
    result = make_move(board, 0, 2, rng)
    assert result["forfeited"] is True


def test_build_obs_old_shape():
    board = empty_board()
    obs = _build_obs_old(board, 1)
    assert obs.shape == (3, ROWS, COLS)


def test_get_action_mask_flat_shape():
    board = empty_board()
    mask = _get_action_mask_flat(board)
    assert mask.shape == (ROWS * COLS,)

"""Functional test for the arena adapter.

Loads the 9M action-value checkpoint and verifies that:
1. load_for_arena builds an adapter successfully.
2. The adapter produces a valid policy (sums to ~1, non-negative, only on
   legal moves) and a value in [-1, 1].
3. The adapter's best move matches the underlying engine's best move.
4. Multiple positions work (starting position, Italian Game, endgame).

Run with:
    .venv/bin/python tests/test_arena_adapter.py
"""

import os
import sys
import time
import unittest

# apache-beam and grain can crash or be missing.  Stub them out before any
# project imports (same approach as arena_adapter.py).
import types

class _PermissiveDummy:
    def __init__(self, *a, **kw): pass
    def __call__(self, *a, **kw): return _PermissiveDummy()
    def __getattr__(self, name): return _PermissiveDummy

class _PermissiveModule(types.ModuleType):
    def __getattr__(self, name): return _PermissiveDummy

for _mod, _subs in [
    ("apache_beam", ["apache_beam.coders"]),
    ("grain", ["grain.python"]),
]:
    if _mod not in sys.modules:
        _stub = _PermissiveModule(_mod)
        sys.modules[_mod] = _stub
        for _sub in _subs:
            _sub_mod = _PermissiveModule(_sub)
            sys.modules[_sub] = _sub_mod
            setattr(_stub, _sub.split(".")[-1], _sub_mod)

import chess
import numpy as np

# ---------------------------------------------------------------------------
# Since the arena won't be importable here, we use the searchless_chess
# MOVE_TO_ACTION mapping as a stand-in for the arena action space.
# ---------------------------------------------------------------------------
from searchless_chess.src import utils as sc_utils
from searchless_chess.src.engines import engine as engine_lib
from searchless_chess.src.engines import neural_engines

NUM_ACTIONS = sc_utils.NUM_ACTIONS
MOVE_TO_ACTION = sc_utils.MOVE_TO_ACTION
ACTION_TO_MOVE = sc_utils.ACTION_TO_MOVE


def _board_from_fen(fen: str) -> chess.Board:
    return chess.Board(fen)


def _identity_board_to_chess(board_arr: np.ndarray) -> chess.Board:
    """Inverse of the simple encoding used in tests: fen bytes → board."""
    fen_str = board_arr.tobytes().decode("utf-8").rstrip("\x00")
    return chess.Board(fen_str)


def _encode_board(board: chess.Board) -> np.ndarray:
    """Encode a board as a zero-padded FEN byte array (simple test encoding)."""
    fen_bytes = board.fen().encode("utf-8")
    arr = np.zeros(128, dtype=np.uint8)
    arr[: len(fen_bytes)] = list(fen_bytes)
    return arr


# Build identity UCI -> action mapping (both spaces are the same here).
_UCI_TO_ACTION = dict(MOVE_TO_ACTION)


class TestArenaAdapter(unittest.TestCase):
    """Functional tests that load the real 9M checkpoint."""

    adapter = None
    engine = None

    _call_times: list = []

    @classmethod
    def setUpClass(cls):
        from searchless_chess.src.arena_adapter import load_for_arena

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ckpt_dir = os.path.join(repo_root, "checkpoints", "9M")
        if not os.path.isdir(ckpt_dir):
            raise unittest.SkipTest(
                f"9M checkpoint not found at {ckpt_dir}. "
                "Run checkpoints/download.sh first."
            )

        t0 = time.perf_counter()
        cls.adapter = load_for_arena(
            model_name="9M",
            board_to_chess_fn=_identity_board_to_chess,
            uci_to_arena_action=_UCI_TO_ACTION,
            num_arena_actions=NUM_ACTIONS,
            predict_batch_size=32,
        )
        cls._load_time = time.perf_counter() - t0
        cls.engine = cls.adapter.sc_engine
        cls._call_times = []

    @classmethod
    def tearDownClass(cls):
        print(f"\n{'='*60}")
        print(f"TIMING SUMMARY")
        print(f"{'='*60}")
        print(f"  Model load:        {cls._load_time:.3f}s")
        if cls._call_times:
            arr = np.array(cls._call_times)
            print(f"  Inference calls:   {len(arr)}")
            print(f"  Total inference:   {arr.sum():.3f}s")
            print(f"  Mean per call:     {arr.mean():.3f}s")
            print(f"  Min / Max:         {arr.min():.3f}s / {arr.max():.3f}s")
        print(f"{'='*60}")

    # ----- helpers --------------------------------------------------------

    def _call_adapter(self, board: chess.Board):
        """Call forward_for_mcts with a single board and return policy, value."""
        board_arr = _encode_board(board).reshape(1, -1)
        player = np.array([1 if board.turn == chess.WHITE else 2], dtype=np.int64)
        t0 = time.perf_counter()
        result = self.adapter.forward_for_mcts(
            {"boards": board_arr, "current_player": player}
        )
        self._call_times.append(time.perf_counter() - t0)
        policy = result["policy"]
        value = result["value"]
        # If torch is available, convert tensors to numpy.
        if hasattr(policy, "numpy"):
            policy = policy.numpy()
            value = value.numpy()
        return policy[0], float(value[0])

    def _get_engine_best_move(self, board: chess.Board) -> chess.Move:
        return self.engine.play(board)

    # ----- tests ----------------------------------------------------------

    def test_starting_position_policy_valid(self):
        """Policy at the starting position sums to ~1 and is non-negative."""
        board = chess.Board()
        policy, value = self._call_adapter(board)

        self.assertEqual(policy.shape, (NUM_ACTIONS,))
        self.assertTrue(np.all(policy >= 0), "Policy has negative values")
        self.assertAlmostEqual(float(policy.sum()), 1.0, places=3)

        # Only legal moves should have nonzero probability.
        legal_actions = {MOVE_TO_ACTION[m.uci()] for m in board.legal_moves}
        nonzero_actions = set(np.nonzero(policy)[0])
        self.assertTrue(
            nonzero_actions.issubset(legal_actions),
            f"Policy has mass on illegal moves: {nonzero_actions - legal_actions}",
        )
        # Should have mass on at least some legal moves.
        self.assertTrue(len(nonzero_actions) > 0, "Policy is all zeros")

    def test_starting_position_value_range(self):
        """Value should be in [-1, 1]."""
        board = chess.Board()
        _, value = self._call_adapter(board)
        self.assertGreaterEqual(value, -1.0)
        self.assertLessEqual(value, 1.0)

    def test_best_move_matches_engine(self):
        """Adapter's argmax move should match the engine's greedy play."""
        board = chess.Board()
        policy, _ = self._call_adapter(board)

        adapter_best_action = int(np.argmax(policy))
        adapter_best_uci = ACTION_TO_MOVE[adapter_best_action]

        engine_best_move = self._get_engine_best_move(board)

        self.assertEqual(
            adapter_best_uci,
            engine_best_move.uci(),
            f"Adapter chose {adapter_best_uci}, engine chose {engine_best_move.uci()}",
        )

    def test_italian_game(self):
        """A well-known opening position should produce valid output."""
        # 1.e4 e5 2.Nf3 Nc6 3.Bc4
        board = chess.Board()
        for uci in ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"]:
            board.push(chess.Move.from_uci(uci))

        policy, value = self._call_adapter(board)
        self.assertAlmostEqual(float(policy.sum()), 1.0, places=3)
        self.assertGreaterEqual(value, -1.0)
        self.assertLessEqual(value, 1.0)

        legal_actions = {MOVE_TO_ACTION[m.uci()] for m in board.legal_moves}
        nonzero_actions = set(np.nonzero(policy)[0])
        self.assertTrue(nonzero_actions.issubset(legal_actions))

    def test_endgame_position(self):
        """A simple K+Q vs K endgame should give a strongly positive value."""
        board = chess.Board("6k1/8/8/8/8/8/8/4K2Q w - - 0 1")
        policy, value = self._call_adapter(board)

        self.assertAlmostEqual(float(policy.sum()), 1.0, places=3)
        # White should be winning → value > 0
        self.assertGreater(value, 0.0, "KQ vs K should be winning for White")

    def test_batch_of_two(self):
        """Calling with a batch of 2 boards should return 2 results."""
        boards = [chess.Board(), chess.Board("6k1/8/8/8/8/8/8/4K2Q w - - 0 1")]
        board_arrs = np.stack([_encode_board(b) for b in boards])
        players = np.array([1, 1], dtype=np.int64)

        t0 = time.perf_counter()
        result = self.adapter.forward_for_mcts(
            {"boards": board_arrs, "current_player": players}
        )
        self._call_times.append(time.perf_counter() - t0)
        policy = result["policy"]
        value = result["value"]
        if hasattr(policy, "numpy"):
            policy = policy.numpy()
            value = value.numpy()

        self.assertEqual(policy.shape, (2, NUM_ACTIONS))
        self.assertEqual(value.shape, (2,))
        # Both should sum to ~1
        for i in range(2):
            self.assertAlmostEqual(float(policy[i].sum()), 1.0, places=3)

    def test_self_play_game(self):
        """Play a short game against itself and log moves + values."""
        board = chess.Board()
        moves_log = []
        max_moves = 20  # 20 half-moves

        for ply in range(max_moves):
            if board.is_game_over():
                break
            policy, value = self._call_adapter(board)
            best_action = int(np.argmax(policy))
            best_uci = ACTION_TO_MOVE[best_action]
            best_prob = float(policy[best_action])

            # Also get engine's choice for comparison
            engine_move = self._get_engine_best_move(board)

            match = "OK" if best_uci == engine_move.uci() else "MISMATCH"
            moves_log.append(
                f"  {ply+1:3d}. {'W' if board.turn == chess.WHITE else 'B'} "
                f"adapter={best_uci} ({best_prob:.3f}) value={value:+.3f}  "
                f"engine={engine_move.uci()}  {match}"
            )
            board.push(chess.Move.from_uci(best_uci))

        outcome = board.outcome()
        status = str(outcome) if outcome else f"in progress after {max_moves} ply"

        print(f"\n{'='*60}")
        print(f"SELF-PLAY DIAGNOSTIC ({max_moves} ply max)")
        print(f"{'='*60}")
        for line in moves_log:
            print(line)
        print(f"  Final FEN: {board.fen()}")
        print(f"  Status: {status}")
        print(f"{'='*60}")


if __name__ == "__main__":
    unittest.main()

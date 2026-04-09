"""Adapter to use searchless_chess models in a torch-based chess arena.

Wraps a JAX-based searchless_chess engine so it presents the
``forward_for_mcts(batch) -> {"policy": Tensor, "value": Tensor}``
interface that MCTS-based arenas expect from a ``torch.nn.Module``.

Example usage from the arena repo::

    from searchless_chess.src.arena_adapter import load_for_arena
    from src.nn.kernels.chess_logic import board_to_chess, move_to_action
    import chess

    # Build UCI -> arena action mapping
    uci_to_arena = {}
    for uci_str in searchless_utils.MOVE_TO_ACTION:
        try:
            arena_idx = move_to_action(chess.Move.from_uci(uci_str))
            uci_to_arena[uci_str] = arena_idx
        except Exception:
            pass

    adapter = load_for_arena(
        model_name="9M",
        board_to_chess_fn=board_to_chess,
        uci_to_arena_action=uci_to_arena,
        num_arena_actions=NUM_ACTIONS,  # arena's action space size
        logic=chess_logic,              # arena's game logic module
    )

    # Use in the arena like any other model
    models["searchless_9M"] = adapter
"""

import os
import sys
from typing import Callable, Dict, Optional

# Stub out apache_beam before any transitive import from searchless_chess
# can trigger it — the C++ mutex implementation crashes on macOS.
if "apache_beam" not in sys.modules:
    import types as _types

    _beam = _types.ModuleType("apache_beam")
    _beam.coders = _types.ModuleType("apache_beam.coders")  # type: ignore[attr-defined]
    sys.modules.setdefault("apache_beam", _beam)
    sys.modules.setdefault("apache_beam.coders", _beam.coders)
    del _beam, _types

import chess
import haiku as hk
import jax
import numpy as np
import scipy.special
from jax import random as jrandom

from searchless_chess.src import tokenizer
from searchless_chess.src import training_utils
from searchless_chess.src import transformer
from searchless_chess.src import utils as sc_utils
from searchless_chess.src.engines import engine as engine_lib
from searchless_chess.src.engines import neural_engines

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Model configurations (mirrors engines/constants.py)
# ---------------------------------------------------------------------------

_MODEL_CONFIGS = {
    "9M": dict(
        policy="action_value",
        num_layers=8, embedding_dim=256, num_heads=8, step=6_400_000,
    ),
    "9M_state_value": dict(
        policy="state_value",
        num_layers=8, embedding_dim=256, num_heads=8, step=-1,
    ),
    "9M_behavioral_cloning": dict(
        policy="behavioral_cloning",
        num_layers=8, embedding_dim=256, num_heads=8, step=-1,
    ),
    "136M": dict(
        policy="action_value",
        num_layers=8, embedding_dim=1024, num_heads=8, step=6_400_000,
    ),
    "270M": dict(
        policy="action_value",
        num_layers=16, embedding_dim=1024, num_heads=8, step=6_400_000,
    ),
}

NUM_RETURN_BUCKETS = 128


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

def _get_base_class():
    """Return torch.nn.Module if available, otherwise a plain object base."""
    if torch is not None:
        return torch.nn.Module
    return object


class SearchlessChessAdapter(_get_base_class()):
    """Wraps a searchless_chess JAX engine for a torch-based MCTS arena.

    The arena's MCTS calls ``model.forward_for_mcts(batch)`` where *batch*
    contains ``boards`` (flattened board arrays) and ``current_player``.
    This adapter converts those to FEN strings, runs the JAX engine, and
    returns a policy distribution + scalar value in the arena's action space.
    """

    training: bool = False

    def __init__(
        self,
        sc_engine: neural_engines.NeuralEngine,
        board_to_chess_fn: Callable[[np.ndarray], chess.Board],
        uci_to_arena_action: Dict[str, int],
        num_arena_actions: int,
        logic=None,
        hparams=None,
        player_white: int = 1,
        player_black: int = 2,
    ):
        super().__init__()
        self.sc_engine = sc_engine
        self.board_to_chess_fn = board_to_chess_fn
        self.uci_to_arena_action = uci_to_arena_action
        self.num_arena_actions = num_arena_actions
        self.player_white = player_white
        self.player_black = player_black
        # Attributes the arena / MCTS may read.
        self.logic = logic
        self.hparams = hparams

    # ----- torch.nn.Module interface the arena expects --------------------

    def forward_for_mcts(self, batch: dict) -> dict:
        boards = batch["boards"]
        players = batch["current_player"]

        _is_tensor = torch is not None and isinstance(boards, torch.Tensor)
        if _is_tensor:
            boards_np = boards.cpu().numpy()
            players_np = players.cpu().numpy()
            device = boards.device
        else:
            boards_np = np.asarray(boards)
            players_np = np.asarray(players)
            device = None

        B = boards_np.shape[0]
        policies = np.zeros((B, self.num_arena_actions), dtype=np.float32)
        values = np.zeros(B, dtype=np.float32)

        for i in range(B):
            board = self.board_to_chess_fn(boards_np[i])
            expected_turn = (
                chess.WHITE if players_np[i] == self.player_white else chess.BLACK
            )
            if board.turn != expected_turn:
                board.turn = expected_turn
            policies[i], values[i] = self._evaluate_position(board)

        if torch is not None:
            return {
                "policy": torch.tensor(policies, device=device, dtype=torch.float32),
                "value": torch.tensor(values, device=device, dtype=torch.float32),
            }
        return {"policy": policies, "value": values}

    def eval(self):
        self.training = False
        return self

    def train(self, mode: bool = True):
        self.training = mode
        return self

    # ----- Internal -------------------------------------------------------

    def _legal_moves_sorted(self, board: chess.Board):
        return engine_lib.get_ordered_legal_moves(board)

    def _map_to_arena_policy(self, board, win_probs):
        """Map per-legal-move scores to the arena's action space."""
        policy = np.zeros(self.num_arena_actions, dtype=np.float32)
        for j, move in enumerate(self._legal_moves_sorted(board)):
            arena_idx = self.uci_to_arena_action.get(move.uci())
            if arena_idx is not None:
                policy[arena_idx] = win_probs[j]
        total = policy.sum()
        if total > 0:
            policy /= total
        return policy

    def _evaluate_position(self, board: chess.Board):
        eng = self.sc_engine

        if isinstance(eng, neural_engines.ActionValueEngine):
            analysis = eng.analyse(board)
            probs = np.exp(analysis["log_probs"])
            win_probs = np.inner(probs, eng._return_buckets_values)
            policy = self._map_to_arena_policy(board, win_probs)
            # V(s) ≈ max_a Q(s,a), mapped from [0,1] to [-1,1]
            value = float(np.max(win_probs)) * 2.0 - 1.0

        elif isinstance(eng, neural_engines.StateValueEngine):
            analysis = eng.analyse(board)
            # next_log_probs are already flipped (negated value for opponent).
            next_probs = np.exp(analysis["next_log_probs"])
            win_probs = np.inner(next_probs, eng._return_buckets_values)
            policy = self._map_to_arena_policy(board, win_probs)
            # Current position value.
            current_probs = np.exp(analysis["current_log_probs"])
            value = float(np.inner(current_probs, eng._return_buckets_values)) * 2.0 - 1.0

        elif isinstance(eng, neural_engines.BCEngine):
            analysis = eng.analyse(board)
            action_probs = scipy.special.softmax(np.asarray(analysis["log_probs"]))
            policy = self._map_to_arena_policy(board, action_probs)
            # BC has no value head.
            value = 0.0

        else:
            raise TypeError(f"Unsupported engine type: {type(eng)}")

        return policy, value


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------

def load_for_arena(
    model_name: str,
    board_to_chess_fn: Callable[[np.ndarray], chess.Board],
    uci_to_arena_action: Dict[str, int],
    num_arena_actions: int,
    logic=None,
    hparams=None,
    checkpoint_dir: Optional[str] = None,
    checkpoint_step: Optional[int] = None,
    use_ema_params: bool = False,
    predict_batch_size: int = 32,
    player_white: int = 1,
    player_black: int = 2,
) -> SearchlessChessAdapter:
    """Load a searchless_chess model and wrap it for arena use.

    Args:
        model_name: One of ``"9M"``, ``"136M"``, ``"270M"``,
            ``"9M_state_value"``, ``"9M_behavioral_cloning"``.
        board_to_chess_fn: Converts an arena board array to a
            ``chess.Board``.
        uci_to_arena_action: Maps UCI move strings (e.g. ``"e2e4"``) to
            the arena's integer action indices.
        num_arena_actions: Size of the arena's action space.
        logic: The arena's game logic module (set on the adapter so the
            arena's validation passes).
        hparams: Optional hparams object attached to the adapter.
        checkpoint_dir: Path to the checkpoint directory.  Defaults to
            ``checkpoints/{model_name}`` relative to the repo root.
        checkpoint_step: Override checkpoint step (``-1`` for latest).
        use_ema_params: Load exponential-moving-average parameters.
        predict_batch_size: JAX inference batch size.  Higher values
            improve throughput for action-value models that evaluate
            every legal move per position.
        player_white: Arena's integer identifier for White.
        player_black: Arena's integer identifier for Black.
    """
    if model_name not in _MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Choose from {list(_MODEL_CONFIGS.keys())}"
        )

    cfg = _MODEL_CONFIGS[model_name]
    policy = cfg["policy"]
    step = checkpoint_step if checkpoint_step is not None else cfg["step"]

    output_size = (
        sc_utils.NUM_ACTIONS if policy == "behavioral_cloning"
        else NUM_RETURN_BUCKETS
    )

    predictor_config = transformer.TransformerConfig(
        vocab_size=sc_utils.NUM_ACTIONS,
        output_size=output_size,
        pos_encodings=transformer.PositionalEncodings.LEARNED,
        max_sequence_length=tokenizer.SEQUENCE_LENGTH + 2,
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        embedding_dim=cfg["embedding_dim"],
        apply_post_ln=True,
        apply_qk_layernorm=False,
        use_causal_mask=False,
    )

    predictor = transformer.build_transformer_predictor(config=predictor_config)

    if checkpoint_dir is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        checkpoint_dir = os.path.join(repo_root, "checkpoints", model_name)

    params = training_utils.load_parameters(
        params=predictor.initial_params(
            rng=jrandom.PRNGKey(1),
            targets=np.ones((1, 1), dtype=np.uint32),
        ),
        step=step,
        use_ema_params=use_ema_params,
        checkpoint_dir=checkpoint_dir,
    )

    predict_fn = neural_engines.wrap_predict_fn(
        predictor=predictor,
        params=params,
        batch_size=predict_batch_size,
    )

    _, return_buckets_values = sc_utils.get_uniform_buckets_edges_values(
        NUM_RETURN_BUCKETS
    )

    sc_engine = neural_engines.ENGINE_FROM_POLICY[policy](
        return_buckets_values=return_buckets_values,
        predict_fn=predict_fn,
    )

    return SearchlessChessAdapter(
        sc_engine=sc_engine,
        board_to_chess_fn=board_to_chess_fn,
        uci_to_arena_action=uci_to_arena_action,
        num_arena_actions=num_arena_actions,
        logic=logic,
        hparams=hparams,
        player_white=player_white,
        player_black=player_black,
    )

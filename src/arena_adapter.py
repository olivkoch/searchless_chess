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

from __future__ import annotations

import os
import sys
import types as _types
from typing import Callable, Dict, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Stub out training-only native deps BEFORE any transitive import from
# searchless_chess can trigger them.  apache_beam and grain both pull in
# C++ extensions whose abseil mutex implementation deadlocks on macOS
# when loaded alongside PyTorch's libtorch.
# ---------------------------------------------------------------------------

class _PermissiveDummy:
    """Accepts any construction/call pattern and returns itself."""
    def __init__(self, *a, **kw): pass
    def __call__(self, *a, **kw): return _PermissiveDummy()
    def __getattr__(self, name): return _PermissiveDummy()


class _PermissiveModule(_types.ModuleType):
    """Module stub that returns a permissive dummy for any attribute access."""
    def __getattr__(self, name):
        # Let Python/module internals behave normally.
        if name.startswith("__"):
            raise AttributeError(name)
        return _PermissiveDummy()  # instance, not class

for _mod_name, _sub_names in [
    ("apache_beam", ["apache_beam.coders"]),
    ("grain", ["grain.python"]),
]:
    if _mod_name not in sys.modules:
        _stub = _PermissiveModule(_mod_name)
        _stub.__file__ = "<stub>"
        sys.modules[_mod_name] = _stub
        for _sub_name in _sub_names:
            _sub_mod = _PermissiveModule(_sub_name)
            _sub_mod.__file__ = "<stub>"
            sys.modules[_sub_name] = _sub_mod
            setattr(_stub, _sub_name.split(".")[-1], _sub_mod)

del _PermissiveModule, _types

# Prevent grpcio's abseil C++ mutex deadlock on macOS.
os.environ.setdefault("GRPC_ENABLE_FORK_SUPPORT", "0")

import chess  # pure Python — safe at module level

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
        position_histories = batch.get("position_histories")
        move_histories = batch.get("move_histories")
        opening_fens = batch.get("opening_fens")

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
            # Reconstruct via move history when available so the FEN
            # includes the correct fullmove_number (the DM tokenizer
            # encodes it, and board_to_chess always sets it to 1).
            if (
                move_histories is not None
                and opening_fens is not None
                and opening_fens[i] is not None
            ):
                board = chess.Board(opening_fens[i])
                for uci in move_histories[i]:
                    board.push_uci(uci)
            else:
                board = self.board_to_chess_fn(boards_np[i])
                expected_turn = (
                    chess.WHITE if players_np[i] == self.player_white else chess.BLACK
                )
                if board.turn != expected_turn:
                    board.turn = expected_turn
            hist_i = position_histories[i] if position_histories is not None else None
            policies[i], values[i] = self._evaluate_position(board, hist_i)

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
        from searchless_chess.src.engines import engine as engine_lib
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

    def _map_to_arena_policy_onehot(self, board, win_probs):
        """One-hot policy: all mass on the argmax move (DM's sorted order).

        DM's ``play()`` calls ``np.argmax(win_probs)`` over sorted legal
        moves.  Normalising and re-indexing into the arena action space can
        change which move wins on near-ties because ``np.argmax`` breaks
        ties by choosing the lowest index.  A one-hot avoids that.
        """
        best_idx = int(np.argmax(win_probs))
        sorted_moves = self._legal_moves_sorted(board)
        best_uci = sorted_moves[best_idx].uci()
        arena_idx = self.uci_to_arena_action.get(best_uci)
        policy = np.zeros(self.num_arena_actions, dtype=np.float32)
        if arena_idx is not None:
            policy[arena_idx] = 1.0
        return policy

    def _apply_repetition_penalty(
        self, board: chess.Board, win_probs: np.ndarray, position_history: dict,
    ) -> None:
        """Clamp win_probs to 0.5 for moves leading to threefold repetition.

        Mirrors DM's ``_update_scores_with_repetitions`` which calls
        ``board.can_claim_threefold_repetition()`` after pushing each
        candidate move.  That python-chess method has two parts:

        (a) The position after our move has occurred 3+ times in the game
            (count in history >= 2).
        (b) The *opponent* has any legal reply that would create a position
            occurring 2+ times total (count in history >= 1), meaning the
            opponent could immediately claim a draw.

        We replicate both checks using the arena's position_history dict.
        """
        if self.logic is None:
            return
        chess_to_board_fn = getattr(self.logic, "chess_to_board", None)
        position_hash_fn = getattr(self.logic, "position_hash", None)
        if chess_to_board_fn is None or position_hash_fn is None:
            return
        from searchless_chess.src.engines import engine as engine_lib
        sorted_legal_moves = engine_lib.get_ordered_legal_moves(board)
        for i, move in enumerate(sorted_legal_moves):
            board.push(move)
            arr_m = chess_to_board_fn(board)
            key_m = position_hash_fn(arr_m)
            count_m = position_history.get(key_m, 0)

            clamped = False
            # Part (a): position after our move already seen 2+ times → 3rd
            if count_m >= 2:
                clamped = True
            else:
                # Part (b): opponent has any reply creating is_repetition(2)
                for opp_move in board.legal_moves:
                    board.push(opp_move)
                    arr_w = chess_to_board_fn(board)
                    key_w = position_hash_fn(arr_w)
                    count_w = position_history.get(key_w, 0)
                    # If opponent's reply lands on the same position as our
                    # push, count the virtual occurrence from our push.
                    if key_w == key_m:
                        count_w += 1
                    if count_w >= 2:  # is_repetition(2): 2 prev + 1 now = 3
                        clamped = True
                        board.pop()
                        break
                    board.pop()

            if clamped:
                win_probs[i] = 0.5
            board.pop()

    def _evaluate_position(self, board: chess.Board, position_history=None):
        from searchless_chess.src.engines import neural_engines
        import scipy.special

        # Terminal position — no legal moves, nothing to evaluate.
        # Keep claimable draws non-terminal for arena play. The arena environment
        # uses its own repetition handling and still expects a move here.
        if board.is_game_over(claim_draw=False):
        # if board.is_game_over(claim_draw=True):
            policy = np.zeros(self.num_arena_actions, dtype=np.float32)
            if board.is_checkmate():
                value = -1.0  # side to move is checkmated
            else:
                value = 0.0  # draw (stalemate, repetition, 50-move, insufficient)
            return policy, value

        eng = self.sc_engine

        if isinstance(eng, neural_engines.ActionValueEngine):
            analysis = eng.analyse(board)
            probs = np.exp(analysis["log_probs"])
            win_probs = np.inner(probs, eng._return_buckets_values)
            if position_history is not None:
                self._apply_repetition_penalty(board, win_probs, position_history)
            policy = self._map_to_arena_policy_onehot(board, win_probs)
            # V(s) ≈ max_a Q(s,a), mapped from [0,1] to [-1,1]
            value = float(np.max(win_probs)) * 2.0 - 1.0

        elif isinstance(eng, neural_engines.StateValueEngine):
            analysis = eng.analyse(board)
            # next_log_probs are already flipped (negated value for opponent).
            next_probs = np.exp(analysis["next_log_probs"])
            win_probs = np.inner(next_probs, eng._return_buckets_values)
            if position_history is not None:
                self._apply_repetition_penalty(board, win_probs, position_history)
            policy = self._map_to_arena_policy_onehot(board, win_probs)
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
    predict_batch_size: int = 1,
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

    # Heavy imports deferred to here so that merely importing the adapter
    # module does not load JAX / haiku / orbax (whose C++ extensions can
    # deadlock alongside PyTorch on macOS).
    import pathlib

    import jax
    import orbax.checkpoint as ocp
    from jax import random as jrandom
    from searchless_chess.src import tokenizer
    from searchless_chess.src import transformer
    from searchless_chess.src import utils as sc_utils
    from searchless_chess.src.engines import neural_engines

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

    # Inline load_parameters from training_utils to avoid importing it
    # (it uses jax.sharding.PositionalSharding which was removed in newer JAX).
    init_params = predictor.initial_params(
        rng=jrandom.PRNGKey(1),
        targets=np.ones((1, 1), dtype=np.uint32),
    )
    checkpoint_steps = ocp.utils.checkpoint_steps(checkpoint_dir)
    resolved_step = checkpoint_steps[-1] if step == -1 else step
    dir_name = "params_ema" if use_ema_params else "params"
    checkpoint_path = pathlib.Path(checkpoint_dir) / str(resolved_step) / dir_name
    restore_args = ocp.checkpoint_utils.construct_restore_args(init_params)
    checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
    params = checkpointer.restore(checkpoint_path, restore_args=restore_args)

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

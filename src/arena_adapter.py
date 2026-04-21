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
import scipy.special

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
    def __getattr__(self, name): return _PermissiveDummy


class _PermissiveModule(_types.ModuleType):
    """Module stub that returns a permissive dummy for any attribute access."""
    def __getattr__(self, name):
        return _PermissiveDummy


for _mod_name, _sub_names in [
    ("apache_beam", ["apache_beam.coders"]),
    ("grain", ["grain.python"]),
]:
    if _mod_name not in sys.modules:
        _stub = _PermissiveModule(_mod_name)
        sys.modules[_mod_name] = _stub
        for _sub_name in _sub_names:
            _sub_mod = _PermissiveModule(_sub_name)
            sys.modules[_sub_name] = _sub_mod
            setattr(_stub, _sub_name.split(".")[-1], _sub_mod)

del _PermissiveModule, _types, _mod_name, _sub_names

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
        debug: bool = False,
        model_name: str = "?",
    ):
        super().__init__()
        self.sc_engine = sc_engine
        self.board_to_chess_fn = board_to_chess_fn
        self.uci_to_arena_action = uci_to_arena_action
        self.num_arena_actions = num_arena_actions
        self.player_white = player_white
        self.player_black = player_black
        self.model_name = model_name
        # Attributes the arena / MCTS may read.
        self.logic = logic
        self.hparams = hparams
        self.debug = debug
        self._call_count = 0

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

        import time as _time
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
            _t0 = _time.perf_counter()
            policies[i], values[i] = self._evaluate_position(board)
            _elapsed = _time.perf_counter() - _t0

            if self.debug and self._call_count < 200:
                best_action = int(np.argmax(policies[i]))
                # Reverse-lookup action -> UCI
                best_uci = "?"
                for uci_str, idx in self.uci_to_arena_action.items():
                    if idx == best_action:
                        best_uci = uci_str
                        break
                import sys
                print(
                    f"[SC_DEBUG {self.model_name} #{self._call_count}] "
                    f"{_elapsed:.3f}s  "
                    f"FEN={board.fen()[:60]}  "
                    f"best={best_uci} (p={policies[i][best_action]:.3f})  "
                    f"v={values[i]:+.3f}  "
                    f"B={B} player={int(players_np[i])}",
                    file=sys.stderr, flush=True,
                )
                self._call_count += 1

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

    def _map_to_arena_policy(self, board, win_probs, temperature=0.01):
        policy = np.zeros(self.num_arena_actions, dtype=np.float32)
        moves = self._legal_moves_sorted(board)
        move_probs = scipy.special.softmax(np.asarray(win_probs) / temperature)
        for j, move in enumerate(moves):
            arena_idx = self.uci_to_arena_action.get(move.uci())
            if arena_idx is not None:
                policy[arena_idx] = move_probs[j]
        return policy

    def _evaluate_position(self, board: chess.Board):
        from searchless_chess.src.engines import neural_engines
        import scipy.special

        eng = self.sc_engine

        if isinstance(eng, neural_engines.ActionValueEngine):
            analysis = eng.analyse(board)
            probs = np.exp(analysis["log_probs"])
            win_probs = np.inner(probs, eng._return_buckets_values)
                
            moves = self._legal_moves_sorted(board)
            for j, move in enumerate(moves):
                board.push(move)
                if board.is_fivefold_repetition() or board.can_claim_threefold_repetition():
                    win_probs[j] = 0.5
                board.pop()

            policy = self._map_to_arena_policy(board, win_probs)
            # V(s) ≈ max_a Q(s,a), mapped from [0,1] to [-1,1]
            value = float(np.max(win_probs)) * 2.0 - 1.0

        elif isinstance(eng, neural_engines.StateValueEngine):
            analysis = eng.analyse(board)
            # next_log_probs are already negated for the opponent by the engine,
            # so win_probs here already ranks moves from the current player's POV.
            next_probs = np.exp(analysis["next_log_probs"])
            win_probs = np.inner(next_probs, eng._return_buckets_values)

            moves = self._legal_moves_sorted(board)
            for j, move in enumerate(moves):
                board.push(move)
                if board.is_fivefold_repetition() or board.can_claim_threefold_repetition():
                    win_probs[j] = 0.5
                board.pop()
            policy = self._map_to_arena_policy(board, win_probs)
            
            # Current position value.
            current_probs = np.exp(analysis["current_log_probs"])
            current_value = float(np.inner(current_probs, eng._return_buckets_values))
            value = current_value * 2.0 - 1.0

        elif isinstance(eng, neural_engines.BCEngine):
            analysis = eng.analyse(board)
            # BC outputs action probabilities directly — these are already from
            # the current player's POV (the model picks the best move to play),
            # so no perspective flip needed.
            action_probs = scipy.special.softmax(np.asarray(analysis["log_probs"]))
            policy = self._map_to_arena_policy(board, action_probs)
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
    debug: bool = False,
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
        debug=debug,
        model_name=model_name,
    )

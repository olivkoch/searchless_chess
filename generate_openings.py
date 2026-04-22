# gen_book.py — regenerate for 1000-game tournament
import os, chess.pgn, numpy as np
import sys
filename = sys.argv[1]
opening_boards = []
with open(filename) as f: # typically, input data/eco_openings.pgn
    while (game := chess.pgn.read_game(f)) is not None:
        opening_boards.append(game.end().board())

rng = np.random.default_rng(seed=1)
n_games = 1000  # your tournament size
opening_indices = rng.choice(
    np.arange(len(opening_boards)),
    size=n_games // 2,
    replace=False,
)
selected = [opening_boards[i] for i in opening_indices]

with open(os.path.expanduser("~/alphazero/dm_book.fen"), "w") as f:
    for b in selected:
        f.write(b.fen() + "\n")

print(f"Wrote {len(selected)} opening FENs")
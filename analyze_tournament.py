import sys
filename = sys.argv[1]
import chess.pgn, collections
stats = collections.Counter()
with open(filename) as f:
    while (g := chess.pgn.read_game(f)):
        w, b, r = g.headers['White'], g.headers['Black'], g.headers['Result']
        if set([w, b]) != {'270M', '9M'}: continue
        if r == '1-0': stats[w + ' wins'] += 1
        elif r == '0-1': stats[b + ' wins'] += 1
        else: stats['draw'] += 1
print(dict(stats))
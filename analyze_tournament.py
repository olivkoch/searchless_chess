import sys
import collections
import chess.pgn

filename = sys.argv[1]

# Collect per-pair stats: (playerA, playerB) -> Counter of outcomes
pair_stats = collections.defaultdict(collections.Counter)

with open(filename) as f:
    while (g := chess.pgn.read_game(f)):
        w, b, r = g.headers['White'], g.headers['Black'], g.headers['Result']
        pair = tuple(sorted([w, b]))
        if r == '1-0':
            pair_stats[pair][w + ' wins'] += 1
        elif r == '0-1':
            pair_stats[pair][b + ' wins'] += 1
        else:
            pair_stats[pair]['draw'] += 1

for pair, stats in sorted(pair_stats.items()):
    total = sum(stats.values())
    print(f'\n{pair[0]} vs {pair[1]} ({total} games):')
    for outcome, count in sorted(stats.items(), key=lambda x: -x[1]):
        print(f'  {outcome}: {count} ({count/total:.0%})')
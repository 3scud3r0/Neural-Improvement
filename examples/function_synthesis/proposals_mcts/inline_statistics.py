def _select(self, node):
    sqrt_parent = math.sqrt(max(1, node.visits))
    exploration = self.exploration
    best = None
    best_score = None
    for child in node.children.values():
        visits = child.visits
        mean = child.score_sum / visits if visits else 0.
        score = mean + exploration * child.prior * sqrt_parent / (1 + visits)
        if best is None or score > best_score:
            best, best_score = child, score
    if best is None:
        raise ValueError('max() arg is an empty sequence')
    return best

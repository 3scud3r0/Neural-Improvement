def _select(self, node):
    sqrt_parent = math.sqrt(max(1, node.visits))
    exploration = self.exploration
    children = iter(node.children.values())
    try:
        best = next(children)
    except StopIteration:
        raise ValueError('max() arg is an empty sequence') from None
    best_score = best.q + exploration * best.prior * sqrt_parent / (1 + best.visits)
    for child in children:
        score = child.q + exploration * child.prior * sqrt_parent / (1 + child.visits)
        if score > best_score:
            best, best_score = child, score
    return best

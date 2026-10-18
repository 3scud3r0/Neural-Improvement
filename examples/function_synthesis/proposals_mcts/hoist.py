def _select(self, node):
    exploration = self.exploration * math.sqrt(max(1, node.visits))
    return max(node.children.values(), key=lambda child:
               child.q + exploration * child.prior / (1 + child.visits))

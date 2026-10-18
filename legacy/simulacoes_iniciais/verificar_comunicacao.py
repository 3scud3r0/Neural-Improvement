"""Teste independente: remover mensagens após os 1000 ciclos de treinamento.
Execute depois do simulacao_coletiva_1000.py.
"""
from pathlib import Path
import torch
from simulacao_coletiva_1000 import Collective, make_batch, EVAL_SIZE, SEED

root = Path(__file__).resolve().parent
checkpoint = torch.load(root/'pesos_1000_ciclos.pt', map_location='cpu', weights_only=True)
network = Collective(communicate=True)
network.load_state_dict(checkpoint['com'])
network.eval()

# Reconstrói os mesmos exemplos de teste independentes do treinamento.
x, y = make_batch(EVAL_SIZE, torch.Generator().manual_seed(SEED+2))
with torch.no_grad():
    prediction_connected = network(x).sigmoid()
    network.communicate = False  # Mantém os pesos, mas remove as mensagens.
    prediction_blocked = network(x).sigmoid()

accuracy = lambda output: ((output >= 0.5) == y.bool()).float().mean().item()
print('Com comunicação: {:.2%}'.format(accuracy(prediction_connected)))
print('Após bloquear mensagens: {:.2%}'.format(accuracy(prediction_blocked)))

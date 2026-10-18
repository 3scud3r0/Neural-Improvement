"""100 redes de 100 neuronios: deteccao coletiva de falhas correlacionadas.
Experimento sintético reprodutível, 1000 ciclos de aprendizado.

Instale: pip install numpy torch matplotlib
Execute: python simulacao_coletiva_1000.py
"""
import csv
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

SEED = 20261017
NUM_NETWORKS = 100
NEURONS = 100
INPUT_DIM = 3
TRAIN_STEPS = 1000
BATCH_SIZE = 64
EVAL_SIZE = 4000
LEARNING_RATE = 0.008
ANOMALOUS_SENSORS = 30
COMMON_BACKGROUND = 0.12
CORRELATED_SIGNAL = 0.72
DEVICE = 'cpu'
OUT_DIR = Path('/mnt/data')

def setup():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)


def make_batch(batch_size: int, generator: torch.Generator):
    """Constrói dados com a mesma distribuição local em ambas as classes.

    y=0: ruído comum de ambiente + ruído privado de cada sensor.
    y=1: o mesmo + um desvio sincronizado em 30 sensores aleatórios.
    Cada sensor, isoladamente, vê exatamente N(0,1) em seu canal principal.
    """
    b, n = batch_size, NUM_NETWORKS
    y = torch.randint(0, 2, (b,), generator=generator).float()
    common = torch.randn(b, 1, generator=generator)
    hidden_fault = torch.randn(b, 1, generator=generator)
    private = torch.randn(b, n, generator=generator)

    # Seleção aleatória de 30 nós distintos por exemplo, somente classe positiva.
    ranking = torch.rand(b, n, generator=generator).argsort(dim=1)
    chosen = torch.zeros(b, n)
    chosen.scatter_(1, ranking[:, :ANOMALOUS_SENSORS], 1.0)
    chosen *= y[:, None]
    correlated_var = CORRELATED_SIGNAL * chosen
    # Variância total exatamente 1, independentemente do rótulo.
    main = (COMMON_BACKGROUND**0.5)*common + (correlated_var**0.5)*hidden_fault + (1-COMMON_BACKGROUND-correlated_var).sqrt()*private

    # Referência ambiental: captura parcialmente o fator comum sem indicar o rótulo.
    environment = (0.65**0.5)*common + (0.35**0.5)*torch.randn(b, n, generator=generator)
    distractor = torch.randn(b, n, generator=generator)
    x = torch.stack((main, environment, distractor), dim=-1)
    return x, y


class Collective(nn.Module):
    """100 redes independentes com 100 neurônios tanh cada uma.

    Cada rede envia seu vetor de 100 ativações a todas as outras 99.
    Receptores agregam a média das 99 mensagens, depois calculam seu voto.
    As matrizes e as portas são particulares de cada rede e aprendidas.
    """
    def __init__(self, communicate: bool):
        super().__init__()
        self.communicate = communicate
        shared = torch.randn(INPUT_DIM, NEURONS) * 0.16
        self.encoder = nn.Parameter(shared[None].expand(NUM_NETWORKS,-1,-1).clone() + torch.randn(NUM_NETWORKS,INPUT_DIM,NEURONS)*0.02)
        self.bias = nn.Parameter(torch.zeros(NUM_NETWORKS, NEURONS))
        self.local_gate = nn.Parameter(torch.ones(NUM_NETWORKS,NEURONS))
        self.msg_gate = nn.Parameter(torch.ones(NUM_NETWORKS,NEURONS)*0.8)
        self.interaction_gate = nn.Parameter(torch.ones(NUM_NETWORKS,NEURONS)*0.7)
        self.hidden_bias = nn.Parameter(torch.zeros(NUM_NETWORKS,NEURONS))
        self.readout = nn.Parameter(torch.randn(NUM_NETWORKS,NEURONS)*0.04)
        self.readout_bias = nn.Parameter(torch.zeros(NUM_NETWORKS))
        self.output_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, return_activity=False):
        h = torch.tanh(torch.einsum('bni,nih->bnh', x, self.encoder) + self.bias)
        if self.communicate:
            msg = (h.sum(dim=1,keepdim=True)-h)/(NUM_NETWORKS-1)
        else:
            msg = torch.zeros_like(h)
        # Coordenação neuronal mediante mensagens dos 99 outros agentes.
        combined = (self.local_gate*h + self.msg_gate*msg + self.interaction_gate*(h*msg)+self.hidden_bias)
        state = torch.tanh(combined)
        local_votes = (state*self.readout).sum(dim=-1)/NEURONS**0.5+self.readout_bias
        collective_logit = local_votes.mean(dim=1) + self.output_bias
        if return_activity:
            return collective_logit, h.detach(), state.detach()
        return collective_logit


def evaluate(models, x, labels):
    outputs={}
    for name,model in models.items():
        model.eval()
        with torch.no_grad():
            scores=torch.cat([model(xi).sigmoid() for xi in x.split(250)],dim=0).numpy()
        y=labels.numpy().astype(int)
        pred=scores>=0.5
        accuracy=float((pred==y).mean())
        pos=scores[y==1];neg=scores[y==0]
        # AUC por ranks, com tratamento de empates via ranks scipy-free.
        order=np.argsort(scores,kind='mergesort')
        sorted_scores=scores[order]
        ranks=np.empty(len(scores),dtype=float)
        j=0
        while j<len(scores):
            k=j+1
            while k<len(scores) and sorted_scores[k]==sorted_scores[j]:k+=1
            ranks[order[j:k]]=(j+1+k)/2
            j=k
        auc=(ranks[y==1].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))
        outputs[name]={'accuracy':accuracy,'auc':float(auc),'mean_positive_score':float(pos.mean()),'mean_negative_score':float(neg.mean()),'scores':scores}
    return outputs


def main():
    setup()
    g_train=torch.Generator().manual_seed(SEED+1)
    g_test=torch.Generator().manual_seed(SEED+2)
    x_test,y_test=make_batch(EVAL_SIZE,g_test)
    connected=Collective(True)
    standalone=Collective(False)
    standalone.load_state_dict(connected.state_dict())
    models={'Com comunicação':connected,'Sem comunicação':standalone}
    optimizers={name:torch.optim.Adam(model.parameters(),lr=LEARNING_RATE) for name,model in models.items()}
    logs=[]
    for step in range(1,TRAIN_STEPS+1):
        x,y=make_batch(BATCH_SIZE,g_train)
        row={'ciclo':step}
        for name,model in models.items():
            model.train()
            optimizer=optimizers[name]
            optimizer.zero_grad(set_to_none=True)
            logits=model(x)
            loss=F.binary_cross_entropy_with_logits(logits,y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=5)
            optimizer.step()
            key='com' if model.communicate else 'sem'
            row['perda_'+key]=round(loss.item(),6)
            row['acerto_treino_'+key]=round(((logits.detach()>=0)==y.bool()).float().mean().item(),6)
        logs.append(row)
        if step in (1,50,100,200,300,500,700,1000):
            print(f'ciclo={step:4d} loss_com={row["perda_com"]:.4f} loss_sem={row["perda_sem"]:.4f}',flush=True)
    results=evaluate(models,x_test,y_test)
    serial={name:{k:v for k,v in r.items() if k!='scores'} for name,r in results.items()}
    print('RESULTADOS',json.dumps(serial,ensure_ascii=False,indent=2),flush=True)
    # Confirma que cada rede possui 100 neurônios e recebe a média exata de 99 outras.
    with torch.no_grad():
        _,h,s=connected(x_test[:8],True)
    assert h.shape==(8,100,100) and s.shape==(8,100,100)
    fast=(h.sum(1,keepdim=True)-h)/99
    slow=torch.stack([h[:,torch.arange(100)!=i,:].mean(1) for i in range(100)],dim=1)
    assert torch.allclose(fast,slow,atol=1e-6)
    # Avaliação de incerteza: bootstrap pareado de acurácia e AUC.
    rng=np.random.default_rng(SEED+3)
    yc=y_test.numpy().astype(bool)
    c=results['Com comunicação']['scores'];s=results['Sem comunicação']['scores']
    differences=[]
    for _ in range(1000):
        idx=rng.integers(0,len(yc),size=len(yc))
        differences.append(((c[idx]>=0.5)==yc[idx]).mean()-((s[idx]>=0.5)==yc[idx]).mean())
    serial['comparacao']={'delta_acuracia':results['Com comunicação']['accuracy']-results['Sem comunicação']['accuracy'],
                          'ic95_bootstrap_delta_acuracia':[float(q) for q in np.percentile(differences,[2.5,97.5])],
                          'total_ciclos':TRAIN_STEPS,'amostras_teste':EVAL_SIZE,'mensagens_dirigidas_por_ciclo':9900,'valores_transmitidos_por_ciclo_sem_lote':990000,
                          'mensagens_dirigidas_durante_treino':9900000,'observacao':'Contagens por exemplo; durante treinamento com lote de 64 multiplicar pelo tamanho do lote.'}
    (OUT_DIR/'resultados_1000_ciclos.json').write_text(json.dumps(serial,indent=2,ensure_ascii=False),encoding='utf-8')
    with (OUT_DIR/'treinamento_1000_ciclos.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(logs[0]));writer.writeheader();writer.writerows(logs)
    np.savez_compressed(OUT_DIR/'teste_1000_ciclos.npz',rotulos=y_test.numpy(),prob_comunicacao=c,prob_sem_comunicacao=s)
    torch.save({'config':{'redes':NUM_NETWORKS,'neuronios':NEURONS,'ciclos':TRAIN_STEPS,'semente':SEED},'com':connected.state_dict(),'sem':standalone.state_dict()},OUT_DIR/'pesos_1000_ciclos.pt')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(12.7,8.2),layout='constrained')
    t=np.array([r['ciclo'] for r in logs]);win=25
    for name,color,suf in [('Com comunicação','#146fa8','com'),('Sem comunicação','#dd8041','sem')]:
        losses=np.array([r['perda_'+suf] for r in logs]);accuracy=np.array([r['acerto_treino_'+suf] for r in logs])
        smooth=lambda v:np.convolve(np.pad(v,(win-1,0),mode='edge'),np.ones(win)/win,mode='valid')
        axes[0,0].plot(t,smooth(losses),label=name,color=color)
        axes[0,1].plot(t,smooth(accuracy),label=name,color=color)
    axes[0,0].set(title='Erro durante os 1.000 ciclos',xlabel='Ciclo de treinamento',ylabel='Entropia cruzada (menor = melhor)')
    axes[0,1].set(title='Acerto de treinamento (média móvel: 25)',xlabel='Ciclo de treinamento',ylabel='Proporção de acertos',ylim=(0.3,1.02))
    axes[0,0].legend()
    names=list(models)
    xs=np.arange(2)
    axes[1,0].bar(xs-0.16,[results[n]['accuracy'] for n in names],width=.32,label='Acurácia',color='#277ca7')
    axes[1,0].bar(xs+0.16,[results[n]['auc'] for n in names],width=.32,label='AUC',color='#8b6bb4')
    axes[1,0].set(xticks=xs,xticklabels=['Com comunicação','Sem comunicação'],ylim=(0,1),title=f'Teste independente: {EVAL_SIZE:,} exemplos'.replace(',','.'),ylabel='Resultado (maior = melhor)')
    axes[1,0].legend()
    axes[1,1].hist(c[~yc],bins=np.linspace(0,1,36),alpha=.7,label='Normal',color='#277ca7',density=True)
    axes[1,1].hist(c[yc],bins=np.linspace(0,1,36),alpha=.55,label='Falha coordenada',color='#d98243',density=True)
    axes[1,1].set(title='Probabilidades da rede coletiva no teste',xlabel='Probabilidade prevista de falha',ylabel='Densidade')
    axes[1,1].legend()
    fig.suptitle('100 redes × 100 neurônios × 1.000 ciclos | detecção distribuída de falhas',fontsize=14,fontweight='bold')
    fig.savefig(OUT_DIR/'grafico_1000_ciclos.png',dpi=165)
    plt.close(fig)
    print('Verificações: dimensões e equivalência entre mensagens explícitas/otimizadas OK',flush=True)

if __name__=='__main__':main()

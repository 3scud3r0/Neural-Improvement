"""Comparacao: 100 redes com H=100,1000,10000 neuronios por rede.
Comunicação aprendida de 4 valores por agente, memória temporal e tarefa
sintética de falha correlacionada em sensores. Cada passo otimiza um minibatch.
Ex.: python experimento_especializacao.py --widths 100 1000 10000 --steps 1000
"""
import argparse, csv, json, math, time, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

N=100; T=3; F_IN=3; MSG=4
SEED=20261018
OUT=Path('resultados')

def setup():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(4)

@torch.no_grad()
def make_batch(b,gen,shift=False):
    """Binario: etiqueta 1 muda covariancia entre sensores (nao marginais).
    Episodios de 3 tempos, ambientes variaveis; teste shift com ruido de sensor maior.
    Cada serie temporal individual tem a mesma distribuicao marginal em y=0 e y=1.
    """
    y=torch.randint(0,2,(b,),generator=gen).float()
    # Amostras temporais independentes, mas localmente com variancia constante.
    base=torch.randn((b,T,1),generator=gen)
    hidden=torch.randn((b,T,1),generator=gen)
    noise=torch.randn((b,T,N),generator=gen)
    rank=torch.rand(b,N,generator=gen).argsort(-1)
    chosen=torch.zeros(b,N)
    chosen.scatter_(1,rank[:,:30],1.)
    chosen*=y[:,None]
    # Forca da falha varia, amostrada no nivel do episodio, sem mudar marginal.
    strength=(0.45+0.30*torch.rand(b,1,generator=gen)) if not shift else (0.38+0.23*torch.rand(b,1,generator=gen))
    corr=chosen*strength
    common=0.10 if not shift else 0.17
    main=math.sqrt(common)*base+torch.sqrt(corr[:,None,:])*hidden+torch.sqrt(1.-common-corr[:,None,:])*noise
    # Canal auxiliar de referencias ambientais, estrutura temporal variavel.
    reference=math.sqrt(0.65)*base+math.sqrt(0.35)*torch.randn(b,T,N,generator=gen)
    distractor=torch.randn(b,T,N,generator=gen)
    x=torch.stack((main,reference,distractor),dim=-1)
    return x,y

class Collective(nn.Module):
    def __init__(self,width,communicate=True):
        super().__init__()
        self.width=width; self.communicate=communicate
        H=width; scale=1/math.sqrt(H)
        # Matrizes distintas por rede, camada H tanh por agente.
        shared_encoder=torch.randn(F_IN,H)*0.14
        self.encoder=nn.Parameter(shared_encoder[None].expand(N,-1,-1).clone()+torch.randn(N,F_IN,H)*0.012)
        self.bias=nn.Parameter(torch.zeros(N,H))
        # 4 dimensoes de mensagem aprendidas por agente
        shared_sender=torch.randn(H,MSG)*scale
        self.sender=nn.Parameter(shared_sender[None].expand(N,-1,-1).clone()+torch.randn(N,H,MSG)*scale*0.1)
        self.receiver=nn.Parameter(torch.randn(N,MSG,H)*0.07)
        # projecao aprendida para atualizar memoria temporal
        self.memory_read=nn.Parameter(shared_sender[None].expand(N,-1,-1).clone()+torch.randn(N,H,MSG)*scale*0.1)
        self.memory_write=nn.Parameter(torch.randn(N,MSG,H)*0.07)
        self.memory_bias=nn.Parameter(torch.zeros(N,MSG))
        # Peso de confiabilidade de cada emissor, treinavel
        self.sender_importance=nn.Parameter(torch.zeros(N))
        self.readout=nn.Parameter(torch.randn(N,H)*0.045*scale)
        self.readout_bias=nn.Parameter(torch.zeros(N))
        self.bias_out=nn.Parameter(torch.tensor(0.))
        self.collective_head=nn.Sequential(nn.Linear(MSG*4,24), nn.Tanh(), nn.Linear(24,1))

    def forward(self,x,force_no_message=False, return_diagnostics=False):
        B=x.shape[0];H=self.width
        mem=x.new_zeros(B,N,MSG)
        # Componente principal da mensagem: h -> R^4. Memoria: R^4 ao longo de T=3.
        weights=torch.softmax(self.sender_importance,dim=0)*N
        vote_acc=x.new_zeros(B,N)
        signals=[]
        moments=x.new_zeros(B,MSG*4)
        for t in range(T):
            local=torch.tanh(torch.einsum('bnf,nfh->bnh',x[:,t],self.encoder)+self.bias)
            signal=torch.tanh(torch.einsum('bnh,nhr->bnr',local,self.sender)/2.0)
            if self.communicate and not force_no_message:
                # Emissores distintos: n recebe a media ponderada dos outros 99.
                total=(signal*weights[None,:,None]).sum(dim=1,keepdim=True)
                incoming=(total-weights[None,:,None]*signal)/(N-weights[None,:,None])
            else:
                incoming=torch.zeros_like(signal)
            local_summary=torch.tanh(torch.einsum('bnh,nhr->bnr',local,self.memory_read)/2.0)
            mem=0.65*mem+0.35*torch.tanh(local_summary+incoming+self.memory_bias)
            moment_t=torch.cat(((local_summary*incoming).mean(1),incoming.square().mean(1),local_summary.square().mean(1),(mem*incoming).mean(1)),dim=-1)
            moments=moments+moment_t/T
            message_decode=torch.einsum('bnr,nrh->bnh',incoming,self.receiver)
            mem_decode=torch.einsum('bnr,nrh->bnh',mem,self.memory_write)
            activity=torch.tanh(local+message_decode+mem_decode+0.8*local*message_decode)
            vote=(activity*self.readout).sum(-1)/math.sqrt(H)+self.readout_bias
            vote_acc=vote_acc+vote/T
            if return_diagnostics:signals.append(signal.detach())
        logit=self.collective_head(moments).squeeze(-1)+0.1*vote_acc.mean(dim=1)+self.bias_out
        if return_diagnostics:return logit, torch.stack(signals,dim=1),vote_acc.detach(),mem.detach(),weights.detach()
        return logit

@torch.no_grad()
def evaluate(model, x,y, ablation=False):
    model.eval(); p=[]
    for batch in x.split(8):
        p.append(model(batch,force_no_message=ablation).sigmoid().cpu())
    prob=torch.cat(p).numpy().astype('float64'); label=y.numpy().astype(bool)
    pred=prob>=0.5
    # AUC por ranks por scipy se disponivel
    from scipy.stats import rankdata
    ranks=rankdata(prob,method='average')
    np_=label.sum();nn_=(~label).sum()
    auc=float((ranks[label].sum()-np_*(np_+1)/2)/(np_*nn_))
    return {'accuracy':float((pred==label).mean()),'auc':auc,'bce':float(np.mean(-(label*np.log(np.clip(prob,1e-8,1-1e-8))+(~label)*np.log(np.clip(1-prob,1e-8,1-1e-8))))),'prob':prob}

def run(width, steps, batch, g_train, test_x, test_y, shift_x, shift_y, log_rows, variant='com'):
    torch.manual_seed(SEED+width)
    model=Collective(width, communicate=(variant=='com'))
    active_params=sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.009,weight_decay=0.025)
    start=time.perf_counter()
    for step in range(1,steps+1):
        x,y=make_batch(batch,g_train)
        optimizer.zero_grad(set_to_none=True)
        logits=model(x)
        loss=F.binary_cross_entropy_with_logits(logits,y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        optimizer.step()
        if step==1 or step%20==0 or step==steps:
            log_rows.append({'width':width,'variant':variant,'step':step,'loss':loss.item(),'train_acc':((logits.detach()>=0)==y.bool()).float().mean().item(),'seconds':time.perf_counter()-start})
        if step%200==0 or step==steps:
            print(f'width={width:5d} variant={variant:3s} step={step:4d}/{steps} loss={loss.item():.3f} t={time.perf_counter()-start:.1f}s',flush=True)
    elapsed=time.perf_counter()-start
    normal=evaluate(model,test_x,test_y)
    shifted=evaluate(model,shift_x,shift_y)
    disable=evaluate(model,test_x,test_y,ablation=True) if variant=='com' else None
    with torch.no_grad():
        _,signals,votes,memory,weight=model(test_x[:8],return_diagnostics=True)
        # Heterogeneidade entre emissores: coeficiente de variacao dos pesos de envio.
        w=weight.cpu().numpy()
        specialization={'sender_weight_cv':float(w.std()/w.mean()),'sender_weight_min':float(w.min()),'sender_weight_max':float(w.max()),'sender_weight_gini':float(np.abs(w[:,None]-w[None,:]).sum()/(2*N*N*w.mean())),'memory_abs_mean':float(memory.abs().mean().item()),'signal_abs_mean':float(signals.abs().mean().item())}
    scores={'width':width,'variant':variant,'steps':steps,'batch':batch,'neurons_total':N*width,'parameters_total':active_params,'duration_sec':elapsed,
            'test':{k:v for k,v in normal.items() if k!='prob'},'shift':{k:v for k,v in shifted.items() if k!='prob'},
            'disable_messages':{k:v for k,v in disable.items() if k!='prob'} if disable else None,'specialization':specialization}
    print('EVAL',json.dumps(scores,ensure_ascii=False),flush=True)
    return model,scores,normal['prob']

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--widths',nargs='+',type=int,default=[100,1000,10000])
    p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--batch',type=int,default=12)
    p.add_argument('--test',type=int,default=1200)
    p.add_argument('--solo',action='store_true',help='Treina baseline sem comunicacao para todas as larguras')
    p.add_argument('--solo-small',action='store_true',help='Treina baseline sem comunicacao somente para 100 neuronios')
    p.add_argument('--out',default='resultados')
    args=p.parse_args()
    global OUT;OUT=Path(args.out);OUT.mkdir(parents=True,exist_ok=True)
    setup()
    g1=torch.Generator().manual_seed(SEED+191)
    g2=torch.Generator().manual_seed(SEED+192)
    test_x,test_y=make_batch(args.test,g1)
    shift_x,shift_y=make_batch(args.test,g2,shift=True)
    log_rows=[];scores=[];pred={}
    for width in args.widths:
        for variant in (['com','sem'] if args.solo else (['com','sem'] if args.solo_small and width==100 else ['com'])):
            # Treino equivalente (mesmos lotes entre larguras/variantes)
            gtrain=torch.Generator().manual_seed(SEED+99)
            model,res,prediction=run(width,args.steps,args.batch,gtrain,test_x,test_y,shift_x,shift_y,log_rows,variant)
            scores.append(res)
            pred[f'{width}_{variant}']=prediction
            # Salvar checkpoint por variante
            torch.save({'width':width,'variant':variant,'seed':SEED,'parameters':model.state_dict()},OUT/f'coletivo_{width}_{variant}.pt')
            del model
    with (OUT/'comparacao_ciclos.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(log_rows[0]));writer.writeheader();writer.writerows(log_rows)
    (OUT/'comparacao_resultados.json').write_text(json.dumps({'seed':SEED,'n_networks':N,'time_steps':T,'message_dim':MSG,'steps':args.steps,'data_type':'sintetico','results':scores},indent=2,ensure_ascii=False),encoding='utf-8')
    np.savez_compressed(OUT/'comparacao_predicoes.npz',labels=test_y.numpy(),**pred)
    plt.rcParams.update({'font.size':11,'axes.spines.right':False,'axes.spines.top':False})
    fig,ax=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for width in args.widths:
        for variant in (['com','sem'] if args.solo else (['com','sem'] if args.solo_small and width==100 else ['com'])):
            rec=[r for r in log_rows if r['width']==width and r['variant']==variant]
            ax[0,0].plot([r['step'] for r in rec],[r['loss'] for r in rec],label=f'{width:,} {variant}')
    ax[0,0].set(xlabel='Ciclo',ylabel='Entropia cruzada no lote',title='Erro durante o treinamento');ax[0,0].legend()
    mainrows=[r for r in scores if r['variant']=='com']
    xs=np.arange(len(mainrows));names=[f"{r['width']:,}" for r in mainrows]
    ax[0,1].bar(xs-0.24,[r['test']['accuracy'] for r in mainrows],width=.24,label='Teste usual')
    ax[0,1].bar(xs,[r['shift']['accuracy'] for r in mainrows],width=.24,label='Mudança de ambiente')
    ax[0,1].bar(xs+.24,[r['disable_messages']['accuracy'] for r in mainrows],width=.24,label='Mensagem desligada')
    ax[0,1].set(xticks=xs,xticklabels=names,ylim=(0,1),xlabel='Neurônios por rede',ylabel='Acurácia',title='Generalização e ablação');ax[0,1].legend()
    ax[1,0].plot(names,[r['test']['auc'] for r in mainrows],'o-',label='Teste habitual')
    ax[1,0].plot(names,[r['shift']['auc'] for r in mainrows],'o-',label='Ambiente modificado')
    ax[1,0].set(ylim=(0,1),xlabel='Neurônios por rede',ylabel='AUC',title='Capacidade de discriminação');ax[1,0].legend()
    ax[1,1].bar(names,[r['duration_sec'] for r in mainrows])
    ax[1,1].set(xlabel='Neurônios por rede',ylabel='Segundos',title='Tempo de treinamento por arquitetura')
    fig.suptitle('100 agentes × 1.000 ciclos | mensagens aprendidas + memória temporal',fontsize=15,fontweight='bold')
    fig.savefig(OUT/'comparacao_100_1000_10000.png',dpi=150);plt.close(fig)
    print('SUCESSO TODOS CONCLUIDOS',flush=True)

if __name__=='__main__':main()

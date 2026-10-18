"""Truly arbitrary-precision *small* neural collective, including message flow.

This reference backend uses Decimal for weights, messages and activations,
with manually differentiated 1-step SGD. Very slow at 100x100 and infeasible
for a million neurons/large MCTS; no claims of accuracy improvements.
"""
from __future__ import annotations
from decimal import Decimal,localcontext
from dataclasses import dataclass
import random

@dataclass
class DecimalResult:
    prediction:str
    message_sample:list[str]
    weight_sample:list[str]
    precision_digits:int

class DecimalSociety:
    def __init__(self,agents=4,width=4,digits=80,seed=2026):
        if not 2<=agents<=100 or not 1<=width<=10000 or not 20<=digits<=1000:
            raise ValueError('Check agent/width/digits bounds')
        self.agents=agents;self.width=width;self.digits=digits
        rng=random.Random(seed)
        # Generate REAL digits rather than adding trailing zeros to FP64.
        def randval():
            sign=-1 if rng.random()<.5 else 1
            return Decimal(sign*rng.randrange(10**digits))/(Decimal(10)**digits)
        with localcontext() as ctx:
            ctx.prec=digits
            self.w_in=[[randval() for _ in range(width)] for _ in range(agents)]
            self.w_msg=[[randval() for _ in range(width)] for _ in range(agents)]
            self.w_out=[[randval() for _ in range(width)] for _ in range(agents)]
            self.bias=[[Decimal(0) for _ in range(width)] for _ in range(agents)]
            self.states=[[Decimal(0) for _ in range(width)] for _ in range(agents)]

    @staticmethod
    def tanh(z):
        return Decimal(2)/(Decimal(1)+(-Decimal(2)*z).exp())-Decimal(1)

    def _forward(self,x):
        with localcontext() as ctx:
            ctx.prec=self.digits
            xx=Decimal(str(x))
            prev_means=[sum(row)/Decimal(self.width) for row in self.states]
            new=[];messages=[]
            for a in range(self.agents):
                message=(sum(prev_means)-prev_means[a])/Decimal(self.agents-1)
                messages.append(message)
                new.append([self.tanh(self.w_in[a][k]*xx+self.w_msg[a][k]*message+
                                      self.bias[a][k]) for k in range(self.width)])
            out=sum(new[a][k]*self.w_out[a][k] for a in range(self.agents)
                    for k in range(self.width))/Decimal(self.agents*self.width)
            return out,new,messages,xx

    def forward(self,x=1,advance=True)->DecimalResult:
        with localcontext() as ctx:
            ctx.prec=self.digits
            result,updated,messages,_=self._forward(x)
            if advance:self.states=updated
            return DecimalResult(str(result),[str(m) for m in messages[:3]],
                                 [str(w) for w in self.w_in[0][:3]],self.digits)

    def step(self,x,target,lr='0.01'):
        """Trainable Decimal weights via *truncated* one-step analytic gradients."""
        with localcontext() as ctx:
            ctx.prec=self.digits
            out,new,messages,xx=self._forward(x)
            error=out-Decimal(str(target))
            alpha=Decimal(str(lr))
            normalizer=Decimal(self.agents*self.width)
            for a in range(self.agents):
                for k in range(self.width):
                    wout=self.w_out[a][k]
                    slope=(Decimal(1)-new[a][k]*new[a][k])
                    delta=alpha*error*wout*slope/normalizer
                    self.w_in[a][k]-=delta*xx
                    self.w_msg[a][k]-=delta*messages[a]
                    self.bias[a][k]-=delta
                    self.w_out[a][k]-=alpha*error*new[a][k]/normalizer
            self.states=new
            return {'prediction':str(out),'squared_error':str(error*error)}

"""Provenance-aware SQLite knowledge graph with verified structural analogies.

Embeddings are a deterministic lexical baseline, not foundation-model embeddings.
Multimodal assets are indexed by hash/URI without downloading private data.
Isomorphism is checked on small typed subgraphs; analogies are suggestions and
MUST be validated on target-domain constraints before use.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib,json,re,sqlite3,time,math
import networkx as nx
from networkx.algorithms import isomorphism

DDL='''
CREATE TABLE IF NOT EXISTS concepts (
 id TEXT PRIMARY KEY, type TEXT NOT NULL, label TEXT NOT NULL,
 attributes TEXT NOT NULL, provenance TEXT NOT NULL, verified INTEGER NOT NULL,
 created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS relations (
 source TEXT NOT NULL, destination TEXT NOT NULL, predicate TEXT NOT NULL,
 confidence REAL NOT NULL CHECK(confidence BETWEEN 0 AND 1),
 evidence TEXT NOT NULL, PRIMARY KEY(source,destination,predicate),
 FOREIGN KEY(source) REFERENCES concepts(id),FOREIGN KEY(destination) REFERENCES concepts(id));
CREATE TABLE IF NOT EXISTS artifacts (
 id TEXT PRIMARY KEY, content_sha256 TEXT NOT NULL, mime_type TEXT NOT NULL,
 uri TEXT NOT NULL, provenance TEXT NOT NULL, linked_concept TEXT NOT NULL,
 FOREIGN KEY(linked_concept) REFERENCES concepts(id));
CREATE TABLE IF NOT EXISTS analogies (
 id INTEGER PRIMARY KEY AUTOINCREMENT, source_pattern TEXT NOT NULL,
 target_pattern TEXT NOT NULL, mapping TEXT NOT NULL, status TEXT NOT NULL,
 evidence TEXT NOT NULL, created REAL NOT NULL);
'''

@dataclass
class Analogy:
    mapping:dict[str,str]
    score:float
    source_nodes:list[str]
    target_nodes:list[str]
    status:str='unverified_hypothesis'
    def as_dict(self):return vars(self)


class ConceptualKnowledgeGraph:
    def __init__(self,path:str|Path=':memory:'):
        self.path=str(path)
        if self.path!=':memory:':Path(self.path).parent.mkdir(parents=True,exist_ok=True)
        self.connection=sqlite3.connect(self.path)
        self.connection.execute('PRAGMA foreign_keys=ON')
        self.connection.executescript(DDL)

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def close(self):self.connection.commit();self.connection.close()

    def add_concept(self,id:str,type:str,label:str,attributes:dict|None=None,
                    provenance:str='user',verified:bool=False):
        if not id or not type or not label:raise ValueError('id/type/label required')
        self.connection.execute('''INSERT INTO concepts VALUES(?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET type=excluded.type,label=excluded.label,
             attributes=excluded.attributes,provenance=excluded.provenance,verified=excluded.verified''',
             (id,type,label,json.dumps(attributes or {},sort_keys=True),provenance,int(verified),time.time()))
        self.connection.commit()

    def relate(self,source:str,destination:str,predicate:str,confidence:float=1.,evidence:str=''):
        if not 0<=confidence<=1:raise ValueError('confidence in [0,1]')
        self.connection.execute('''INSERT INTO relations VALUES(?,?,?,?,?)
            ON CONFLICT(source,destination,predicate) DO UPDATE SET
            confidence=excluded.confidence,evidence=excluded.evidence''',
            (source,destination,predicate,confidence,evidence))
        self.connection.commit()

    def register_artifact(self,id:str,uri:str,mime_type:str,sha256:str,
                          linked_concept:str,provenance:str):
        if not re.fullmatch(r'[0-9a-fA-F]{64}',sha256):raise ValueError('SHA256 must be hex')
        if mime_type not in {'text/plain','application/pdf','image/png','image/jpeg',
                             'video/mp4','application/json'}:raise ValueError('unsupported MIME')
        self.connection.execute('INSERT INTO artifacts VALUES(?,?,?,?,?,?)',
                                (id,sha256.lower(),mime_type,uri,provenance,linked_concept))
        self.connection.commit()

    def _graph(self,subset:list[str]|None=None):
        g=nx.DiGraph()
        for row in self.connection.execute('SELECT id,type,label,verified FROM concepts'):
            if subset is None or row[0] in subset:
                g.add_node(row[0],kind=row[1],label=row[2],verified=bool(row[3]))
        for row in self.connection.execute('SELECT source,destination,predicate,confidence FROM relations'):
            if row[0] in g and row[1] in g:
                g.add_edge(row[0],row[1],kind=row[2],confidence=row[3])
        return g

    def lexical_search(self,query:str,limit=10):
        def tokens(t):return set(re.findall(r'[a-zA-ZÀ-ÿ0-9]+',t.lower()))
        q=tokens(query)
        results=[]
        for row in self.connection.execute('SELECT id,type,label,attributes,verified FROM concepts'):
            vocab=tokens(row[2]+' '+row[3])
            score=len(q&vocab)/len(q|vocab) if q|vocab else 0.
            if score>0:results.append({'id':row[0],'type':row[1],'label':row[2],
                                       'verified':bool(row[4]),'lexical_score':score})
        return sorted(results,key=lambda d:d['lexical_score'],reverse=True)[:limit]

    def discover_analogies(self,source_nodes:list[str],target_nodes:list[str],
                           limit=5,ignore_domain_labels=True)->list[Analogy]:
        if not 2<=len(source_nodes)<=12 or not 2<=len(target_nodes)<=12:
            raise ValueError('small patterns of 2..12 nodes required')
        graph=self._graph()
        if any(n not in graph for n in source_nodes+target_nodes):raise ValueError('unknown concept')
        ga=graph.subgraph(source_nodes).copy();gb=graph.subgraph(target_nodes).copy()
        def nodes_match(a,b):
            return ignore_domain_labels or a['kind']==b['kind']
        matcher=isomorphism.DiGraphMatcher(ga,gb,node_match=nodes_match,
                            edge_match=lambda a,b:a['kind']==b['kind'])
        out=[]
        for mapping in matcher.isomorphisms_iter():
            verified=sum(graph.nodes[n]['verified'] for n in mapping)
            score=0.5+0.5*verified/max(1,len(mapping))
            out.append(Analogy(dict(mapping),score,source_nodes,target_nodes))
            if len(out)>=limit:break
        return out

    def preserve_analogy(self,analogy:Analogy,evidence:str=''):
        # Never upgrades a speculative analogy to a verified fact automatically.
        self.connection.execute('''INSERT INTO analogies
             (source_pattern,target_pattern,mapping,status,evidence,created)
             VALUES (?,?,?,?,?,?)''',
             (json.dumps(analogy.source_nodes),json.dumps(analogy.target_nodes),
              json.dumps(analogy.mapping),analogy.status,evidence,time.time()))
        self.connection.commit()

    def stats(self):
        return {table:self.connection.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
                for table in ('concepts','relations','artifacts','analogies')}


def seed_cross_domain_example(graph:ConceptualKnowledgeGraph):
    """Two isomorphic *structural* examples; no physical transfer is asserted."""
    examples={
       'electric': [('voltage','driver'),('resistor','constraint'),('current','flow')],
       'hydraulic': [('pressure','driver'),('pipe','constraint'),('flux','flow')],
    }
    for domain,nodes in examples.items():
        for name,kind in nodes:
            graph.add_concept(f'{domain}:{name}',kind,f'{name} in {domain}',
                              {'domain':domain},provenance='synthetic_illustration')
        prefix=lambda name:f'{domain}:{name}'
        graph.relate(prefix(nodes[0][0]),prefix(nodes[2][0]),'drives',evidence='synthetic analogy')
        graph.relate(prefix(nodes[1][0]),prefix(nodes[2][0]),'constrains',evidence='synthetic analogy')
    a=[f'electric:{name}' for name,_ in examples['electric']]
    b=[f'hydraulic:{name}' for name,_ in examples['hydraulic']]
    return graph.discover_analogies(a,b)

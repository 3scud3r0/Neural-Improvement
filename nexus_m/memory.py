"""Persistent, indexed, verified, replayable experience memory (SQLite WAL).

'verified' means consistent with the examples witnessed, not a proof that the
candidate matches an unknown underlying rule. Source examples and hashes stay.
"""
from __future__ import annotations
import json, sqlite3, hashlib, time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from .grammar import value_on_examples, pretty
from .tasks import Task

DDL='''
CREATE TABLE IF NOT EXISTS episodes (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 fingerprint TEXT NOT NULL,
 family TEXT NOT NULL,
 examples_json TEXT NOT NULL,
 program_json TEXT NOT NULL,
 examples_exact INTEGER NOT NULL,
 score REAL NOT NULL,
 origin TEXT NOT NULL,
 created_at REAL NOT NULL,
 UNIQUE(fingerprint,program_json)
);
CREATE INDEX IF NOT EXISTS idx_family ON episodes(family, score DESC);
CREATE INDEX IF NOT EXISTS idx_fingerprint ON episodes(fingerprint);
CREATE TABLE IF NOT EXISTS concepts (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 canonical_program TEXT NOT NULL UNIQUE,
 program_json TEXT NOT NULL,
 success_count INTEGER NOT NULL DEFAULT 0,
 failure_count INTEGER NOT NULL DEFAULT 0,
 last_used REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS observations (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 timestamp REAL NOT NULL,
 kind TEXT NOT NULL,
 payload_json TEXT NOT NULL
);
'''

class PersistentMemory:
    def __init__(self, path: str | Path):
        self.path=Path(path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.connection=sqlite3.connect(str(self.path),timeout=30)
        self.connection.execute('PRAGMA journal_mode=WAL')
        self.connection.executescript(DDL)

    def close(self):
        self.connection.commit()
        self.connection.close()

    def __enter__(self): return self
    def __exit__(self,*args): self.close()

    def remember(self,task:Task,program:Sequence[str],origin='mcts')->dict:
        prog=list(program)
        score,exact=value_on_examples(prog,task.examples)
        self.connection.execute('''INSERT OR REPLACE INTO episodes
          (fingerprint,family,examples_json,program_json,examples_exact,score,origin,created_at)
          VALUES (?,?,?,?,?,?,?,?)''',
          (task.fingerprint,task.family,json.dumps(task.examples),json.dumps(prog),int(exact),score,origin,time.time()))
        if exact:
            canonical=pretty(prog)
            self.connection.execute('''INSERT INTO concepts
              (canonical_program,program_json,success_count,failure_count,last_used)
              VALUES (?,?,1,0,?) ON CONFLICT(canonical_program)
              DO UPDATE SET success_count=success_count+1,last_used=excluded.last_used''',
              (canonical,json.dumps(prog),time.time()))
        self.connection.commit()
        return {'exact_on_examples':exact,'score':score}

    def retrieve(self, task:Task,limit=12)->list[dict]:
        """Rank candidate programs by CURRENT evidence, never hidden test labels.

        Mix exact-fingerprint episodes and consolidated concepts. Inference
        receives an unlabeled task; no family name is used as a shortcut.
        """
        rows=self.connection.execute("""SELECT program_json,origin FROM episodes
              WHERE examples_exact=1 ORDER BY (fingerprint=?) DESC,id DESC LIMIT 350""",
              (task.fingerprint,)).fetchall()
        concepts=self.connection.execute("""SELECT program_json FROM concepts
              ORDER BY success_count DESC,last_used DESC LIMIT 350""").fetchall()
        rows += [(r[0],'consolidated_concept') for r in concepts]
        seen=set();candidates=[]
        for program_json,origin in rows:
            if program_json in seen:continue
            seen.add(program_json)
            candidate=json.loads(program_json)
            score,exact=value_on_examples(candidate,task.examples)
            candidates.append({'program':candidate,'score':score,'exact':exact,
                               'origin':origin,'family':'unlabeled'})
        # Simplicity is a prior; input-output matches can still be spurious.
        candidates.sort(key=lambda x:(x['exact'],x['score'],-len(x['program'])),reverse=True)
        return candidates[:limit]

    def concepts(self,limit=20)->list[dict]:
        rows=self.connection.execute('''SELECT canonical_program,success_count,failure_count
           FROM concepts ORDER BY success_count DESC,last_used DESC LIMIT ?''',(limit,)).fetchall()
        return [dict(zip(('expression','successes','failures'),row)) for row in rows]

    def stats(self)->dict:
        ep=self.connection.execute('SELECT count(*),sum(examples_exact) FROM episodes').fetchone()
        n=self.connection.execute('SELECT count(*) FROM concepts').fetchone()[0]
        return {'episodes':ep[0], 'example_verified_episodes':ep[1] or 0,'concepts':n}

    def observation(self,kind,payload):
        self.connection.execute('INSERT INTO observations (timestamp,kind,payload_json) VALUES (?,?,?)',
                                (time.time(),kind,json.dumps(payload,ensure_ascii=False,default=str)))
        self.connection.commit()

    def snapshot(self,destination:str|Path):
        with sqlite3.connect(str(destination)) as target:
            self.connection.backup(target)

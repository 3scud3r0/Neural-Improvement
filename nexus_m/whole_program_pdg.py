"""Static Python symbol/call dependencies and coordinated function patches.

This is a best-effort call graph, not a sound whole-language PDG: dynamic
imports, reflection, monkeypatches and general dispatch remain unresolved.
Symbols include full module and lexical scope to avoid basename collisions.
"""
from __future__ import annotations
import ast
from dataclasses import dataclass
import os
from pathlib import Path
import tokenize
from .function_ast import FunctionTarget, parse_proposal


@dataclass
class Symbol:
    name:str
    module:str
    qualified_name:str
    path:Path
    node:ast.AST
    enclosing_functions:tuple[str,...]
    owner_class:str|None


class ProgramDependencyGraph:
    IGNORE={'.git','.venv','venv','__pycache__','.pytest_cache','.mypy_cache',
            '.ruff_cache','node_modules','runs','build','dist'}
    def __init__(self,workspace):
        self.workspace=Path(workspace).resolve()
        self.build_graph()

    def build_graph(self):
        self.symbols={};self.file_map={};self.call_graph={};self.classes=set()
        self.unresolved=[];self.parse_errors=[];self.modules={};self.aliases={};self.exports={}
        files=[]
        for folder,dirs,names in os.walk(self.workspace,followlinks=False):
            dirs[:]=[d for d in sorted(dirs) if d not in self.IGNORE and not (Path(folder)/d).is_symlink()]
            for name in sorted(names):
                path=Path(folder)/name
                if name.endswith('.py') and not path.is_symlink():files.append(path)
        for path in files:
            relative=path.relative_to(self.workspace).with_suffix('')
            parts=list(relative.parts)
            package=parts[-1]=='__init__'
            if package:parts.pop()
            module='.'.join(parts) or '__init__'
            try:
                with tokenize.open(path) as stream:tree=ast.parse(stream.read())
            except (SyntaxError,UnicodeError,OSError) as exc:
                self.parse_errors.append(dict(path=str(relative),error=str(exc)));continue
            self.modules[module]=(path,tree,package)
            aliases={}
            for node in tree.body:
                if isinstance(node,(ast.Import,ast.ImportFrom)):aliases.update(self._imports(node,module,package))
            self.aliases[module]=aliases
            for alias,targets in aliases.items():
                if targets:self.exports[f'{module}.{alias}']=set(targets)
            self._index(tree.body,module,path,(),(),None)
        for info in list(self.symbols.values()):self._calls(info)
        return self

    def _imports(self,node,module,package=False):
        result={}
        if isinstance(node,ast.Import):
            for alias in node.names:
                result[alias.asname or alias.name.split('.')[0]]={alias.name if alias.asname else alias.name.split('.')[0]}
        else:
            parent=module.split('.') if package else module.split('.')[:-1]
            if node.level:
                keep=len(parent)-(node.level-1)
                prefix=parent[:max(0,keep)]
                base='.'.join(prefix+([node.module] if node.module else []))
            else:base=node.module or ''
            for alias in node.names:
                if alias.name=='*':continue
                result[alias.asname or alias.name]={f'{base}.{alias.name}'.strip('.')}
        return result

    def _index(self,body,module,path,scope,enclosing,owner):
        for node in body:
            if isinstance(node,ast.ClassDef):
                name='.'.join((module,*scope,node.name));self.classes.add(name)
                self._index(node.body,module,path,scope+(node.name,),enclosing,name)
            elif isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
                qualified='.'.join((*scope,node.name));name=f'{module}.{qualified}'
                if name in self.symbols:
                    self.parse_errors.append(dict(path=str(path),error=f'ambiguous symbol {name}'))
                    continue
                self.symbols[name]=Symbol(name,module,qualified,path,node,enclosing,owner)
                self.file_map[name]=path
                self._index(node.body,module,path,scope+(node.name,),enclosing+(name,),owner)
            else:self._index(list(ast.iter_child_nodes(node)),module,path,scope,enclosing,owner)

    def _normalize(self,targets):
        output=set();pending=list(targets);seen=set()
        while pending:
            target=pending.pop()
            if target in seen:continue
            seen.add(target)
            if target in self.exports and self.exports[target]!={target}:pending.extend(self.exports[target])
            else:output.add(target)
        return output

    def _calls(self,info):
        graph=self
        aliases={name:set(values) for name,values in self.aliases.get(info.module,{}).items()}
        node=info.node
        for arg in (*node.args.posonlyargs,*node.args.args,*node.args.kwonlyargs):aliases[arg.arg]=set()
        if info.owner_class:
            aliases['self']={info.owner_class};aliases['cls']={info.owner_class}
        def resolve(expression):
            if isinstance(expression,ast.Name):
                if expression.id in aliases:return graph._normalize(aliases[expression.id])
                for prefix in (info.name,*reversed(info.enclosing_functions),info.module):
                    name=prefix+'.'+expression.id
                    if name in graph.symbols or name in graph.classes:return {name}
                import builtins
                if hasattr(builtins,expression.id):return {'builtins.'+expression.id}
                return set()
            if isinstance(expression,ast.Attribute):
                return graph._normalize({name+'.'+expression.attr for name in resolve(expression.value)})
            return set()
        class Calls(ast.NodeVisitor):
            def visit_FunctionDef(self,node):pass
            def visit_AsyncFunctionDef(self,node):pass
            def visit_ClassDef(self,node):pass
            def visit_Import(self,node):aliases.update(graph._imports(node,info.module,graph.modules[info.module][2]))
            def visit_ImportFrom(self,node):aliases.update(graph._imports(node,info.module,graph.modules[info.module][2]))
            def visit_Assign(self,node):
                self.visit(node.value)
                types=resolve(node.value.func)&graph.classes if isinstance(node.value,ast.Call) else set()
                for target in node.targets:
                    if isinstance(target,ast.Name):aliases[target.id]=set(types)
            def visit_Call(self,node):
                targets=resolve(node.func)
                if not targets:graph.unresolved.append(dict(caller=info.name,line=node.lineno,call=ast.unparse(node.func)))
                for callee in targets:graph.call_graph.setdefault(callee,set()).add(info.name)
                self.generic_visit(node)
        visitor=Calls()
        for statement in node.body:visitor.visit(statement)

    def resolve_symbol(self,name):
        if name in self.symbols:return name
        matches=[symbol for symbol in self.symbols if symbol.endswith('.'+name)]
        if len(matches)!=1:raise ValueError(f'missing or ambiguous repository symbol: {name}')
        return matches[0]

    def get_impact_closure(self,function_name):
        target=self.resolve_symbol(function_name);affected={target};pending=[target]
        while pending:
            current=pending.pop()
            for caller in self.call_graph.get(current,()):
                if caller not in affected:affected.add(caller);pending.append(caller)
        return affected

    def impact_report(self,name):
        affected=self.get_impact_closure(name)
        return dict(symbol=self.resolve_symbol(name),affected=sorted(affected),
            files=sorted({str(self.file_map[s].relative_to(self.workspace)) for s in affected}),
            unresolved_calls=self.unresolved,parse_errors=self.parse_errors,
            complete=False,scope='static call dependencies; dynamic Python is not fully resolved')

    def create_multi_file_transaction(self,primary_patch):
        """Render complete module files, rather than replacing files with functions."""
        patches={};changed_interfaces=[]
        for supplied,source in primary_patch.items():
            symbol=self.resolve_symbol(supplied);info=self.symbols[symbol]
            relative=str(info.path.relative_to(self.workspace))
            target=FunctionTarget.read(self.workspace,relative,info.qualified_name)
            if relative in patches:
                raw=patches[relative];text=raw.decode(target.encoding)
                target.original=raw;target.text=text
                from .function_ast import find_function
                target.function=find_function(ast.parse(text,type_comments=True),info.qualified_name)
            candidate=parse_proposal(source)
            def api(node):
                return (type(node).__name__,ast.dump(node.args),
                        tuple(ast.dump(d) for d in node.decorator_list),
                        ast.dump(node.returns) if node.returns is not None else None)
            if api(candidate)!=api(info.node):changed_interfaces.append(symbol)
            patches[relative]=target.render(candidate)
        # A conservative interface-change rule: include all statically resolved
        # callers' files. Tests must still check the actual coordinated behavior.
        for symbol in changed_interfaces:
            files={str(self.file_map[s].relative_to(self.workspace)) for s in self.get_impact_closure(symbol)}
            if not files<=patches.keys():
                raise ValueError(f'interface change lacks affected module patches: {sorted(files-patches.keys())}')
        return patches

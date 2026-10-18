import ast
from itertools import product
import pytest
from nexus_m.typed_program import IntExpr,enumerate_typed,typed_function
from nexus_m.egraph_synthesis import EGraph,ENode,egraph_proposals
from nexus_m.oracle_coevolution import AdversarialOracle,outcome
from nexus_m.levin_universal_search import LevinUniversalSearch,prefix_code


def expr(source,variables=('x',)):
    return IntExpr.from_ast(ast.parse(source,mode='eval').body,variables,builtin_calls=True)


def test_typed_ir_rejects_raw_objects_calls_and_bad_values():
    for source in ('x.real','__import__("os")','1.5+x','True','lambda x:x'):
        with pytest.raises(ValueError):expr(source)
    with pytest.raises(TypeError):expr('x+1').evaluate({'x':True})
    with pytest.raises(RuntimeError):expr('x+1').evaluate({'x':1},fuel=2)
    assert expr('x+1').evaluate({'x':1},fuel=3)==2
    assert expr('x//-2').evaluate({'x':7})==-4


def test_enumeration_is_well_typed_bounded_and_composes_terms():
    terms=list(enumerate_typed(max_nodes=5,max_programs=1000))
    assert len(terms)<=1000 and all(t.nodes<=5 for t in terms)
    assert expr('x*x') in terms
    for term in terms[:100]:compile(ast.Expression(ast.fix_missing_locations(term.to_ast())),'<term>','eval')


def test_egraph_congruence_rebuild_and_cycle_safe_extraction():
    graph=EGraph()
    x=graph.add(ENode('var',value='x'));y=graph.add(ENode('var',value='y'))
    zero=graph.add(ENode('const',value=0))
    a=graph.add(ENode('add',(x,zero)));b=graph.add(ENode('add',(y,zero)))
    graph.union(x,y);graph.rebuild()
    assert graph.find(a)==graph.find(b)
    graph.union(x,a);graph.rebuild()
    assert graph.extract_expr(x).op=='var'  # x == x+0 contains a graph cycle


def test_egraph_preserves_polynomials_and_rejects_float_or_unknown_semantics():
    original=expr('(x*1)+(0*(x+1))')
    graph=EGraph(max_nodes=500);root=graph.from_expr(original)
    stats=graph.saturate(max_iterations=4)
    selected=graph.extract_expr(root)
    assert selected==expr('x')
    assert selected.polynomial()==original.polynomial()
    assert stats['nodes']<=500
    for text in ('x//0','x+1.0','f(x)'):
        with pytest.raises(ValueError):EGraph().from_ast(ast.parse(text,mode='eval').body)
    with pytest.raises(ValueError):graph.saturate(['invented_rule'])
    with pytest.raises(ValueError):graph.extract_expr(root,{'var':0})


def test_egraph_factorization_and_scoped_type_annotations():
    source=expr('x*x+x*2')
    graph=EGraph(600);root=graph.from_expr(source)
    graph.saturate(['commutative','factor'],max_iterations=3)
    assert graph.extract_expr(root).polynomial()==source.polynomial()
    untyped=ast.parse('def f(x): return x+0').body[0]
    assert list(egraph_proposals(untyped,None))==[]
    typed=ast.parse('def f(x:int): return x+0').body[0]
    assert list(egraph_proposals(typed,None))


def test_oracle_distinguishes_proof_falsification_and_finite_evidence():
    oracle=AdversarialOracle()
    same=oracle.compare(expr('(x+1)*(x+1)'),expr('x*x+2*x+1'))
    assert same.proved and same.proof_kind=='integer_polynomial_normal_form'
    bad=oracle.compare(expr('x*x'),expr('0'))
    assert bad.falsified and bad.counterexample is not None
    finite=oracle.compare(expr('x//2'),expr('x//2'),solver=False)
    assert finite.status=='tested_no_counterexample' and not finite.proved
    # Both sides raising the SAME error must not be treated as divergence.
    error=oracle.compare(expr('x//0'),expr('(x+1)//0'),solver=False)
    assert not error.falsified


def test_smt_checks_definedness_and_negative_floor_division():
    pytest.importorskip('z3')
    oracle=AdversarialOracle()
    assert oracle.compare(expr('x//-2'),expr('(-x)//2')).proved
    difference=oracle.compare(expr('x//x'),expr('1'))
    assert difference.falsified and difference.counterexample=={'x':0}
    undefined=oracle.compare(expr('x//0'),expr('(x+1)//0'))
    assert undefined.proved


def test_description_reward_never_rewards_unverified_shortcuts():
    assert AdversarialOracle.calculate_mdl_reward('def f(x):return x+1','def f(x):return True',100.,False)<0
    with pytest.raises(ValueError):AdversarialOracle.calculate_mdl_reward('x=1','x=1',float('nan'),True)


def test_prefix_codes_are_prefix_free_on_a_finite_cross_section():
    terms=list(enumerate_typed(max_nodes=3,max_programs=100))
    codes=[prefix_code(t) for t in terms]
    assert len(set(codes))==len(codes)
    assert not any(a!=b and b.startswith(a) for a,b in product(codes,repeat=2))


def test_dovetailing_reaches_nontrivial_program_instead_of_restarting_prefix():
    search=LevinUniversalSearch(max_depth=5,max_programs=512)
    result=search.search([({'x':x},x*x) for x in (-3,-1,0,2,7)],max_fuel=10000)
    assert result is not None and result.expression==expr('x*x')
    assert result.fuel_spent<=10000 and result.programs_generated>5
    assert result.description_bits==len(prefix_code(result.expression))
    assert any(r['verification_steps'] for r in search.last_statistics.allocations)


def test_dovetailing_stops_on_fuel_and_search_space_exhaustion():
    search=LevinUniversalSearch(max_depth=3,max_programs=100)
    assert search.search([({'x':2},999)],max_fuel=1) is None
    assert search.last_statistics.fuel_spent<=1
    tiny=LevinUniversalSearch(max_depth=1,max_programs=5)
    assert tiny.search([({'x':2},999)],max_fuel=100) is None
    assert tiny.last_statistics.stop_reason=='search_space_exhausted'


def test_vectorized_enumeration_filters_then_scalar_checks():
    search=LevinUniversalSearch(max_depth=3,max_programs=100)
    examples=[({'x':x},x*x) for x in (-3,-1,0,2,7)]
    solution=search.search_batched(examples,device='cpu',batch_size=16,max_instruction_slots=1000)
    assert solution is not None and solution.expression==expr('x*x')
    assert solution.backend=='vectorized_prefilter'
    assert search.last_statistics.vector_instruction_slots<=1000
    assert search.search_batched(examples,device='cpu',max_instruction_slots=10) is None
    assert search.last_statistics.vector_instruction_slots<=10

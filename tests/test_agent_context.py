"""Evidence-selection regressions: provenance, scope and balanced context."""
from src.agents.context import select_agent_context
from src.contracts import Chunk, Evidence, GenerationRequest, UserContext


def hit(doc, text, title='', start=20):
    key=f'{doc}:{start}:{start+len(text)}'
    return Evidence(citation_id=key, chunk=Chunk(chunk_id=key,doc_id=doc,source_type='docs',
        title=title,text=text,start_char=start,end_char=start+len(text)),score=1,retrieval_method='bm25')


def req(items, question='DEK caching policy and key rotation during KMS outage?'):
    return GenerationRequest(question=question,user=UserContext(user_id='test',
        allowed_doc_ids=frozenset(e.chunk.doc_id for e in items)),evidence=tuple(items))


def test_followup_context_keeps_initial_policy_and_new_rotation_evidence():
    initial=hit('policy', 'DEK caching policy: never persist plaintext keys. '*50, 'Key policy')
    noise=hit('marketing', 'Customer-facing materials and enterprise messaging. '*50, 'Messaging')
    useful=hit('rotation','KMS outage and key rotation: bounded retries and dual decrypt. '*50,'Key rotation')
    r=req([initial,noise,useful]);out=select_agent_context(r,3000,{initial.citation_id})
    assert {e.chunk.doc_id for e in out} == {'policy','rotation'}
    assert sum(len(e.chunk.text) for e in out)<=3000
    for e in out:
        source=next(s for s in r.evidence if s.chunk.doc_id==e.chunk.doc_id)
        offset=e.chunk.start_char-source.chunk.start_char
        assert e.chunk.text==source.chunk.text[offset:offset+len(e.chunk.text)]
        assert e.citation_id==f'{e.chunk.doc_id}:{e.chunk.start_char}:{e.chunk.end_char}'


def test_initial_only_keeps_retrieval_order():
    first=hit('a','Other policy');second=hit('b','DEK KMS policy')
    r=req([first,second]);out=select_agent_context(r,100,{first.citation_id,second.citation_id})
    assert out==[first,second]


def test_irrelevant_followups_do_not_displace_initial_fact():
    first=hit('a','DEK caching policy '*50);other=hit('b','Banana bread recipe '*50)
    out=select_agent_context(req([first,other]),500,{first.citation_id})
    assert out[0].chunk.doc_id=='a'


def test_balanced_selection_is_deterministic_and_excludes_nothing_outside_pool():
    items=[hit('a','DEK caching policy '*200),hit('b','KMS outage rotation '*200)]
    r=req(items);a=select_agent_context(r,2000,{items[0].citation_id})
    assert a==select_agent_context(r,2000,{items[0].citation_id})
    assert len({e.citation_id for e in a})==len(a)
    assert all(r.user.permits(e.chunk.doc_id) for e in a)
    assert sum(len(e.chunk.text) for e in a)<=2000


def test_followup_passage_keeps_priority_lead_and_query_dense_tail():
    initial=hit('a', 'compaction metrics '*200)
    text=('Recommended knobs in order: budget_ms_per_step then budget_bytes_per_step. '
          + 'background notes ' * 180 + ' compaction metrics counters disable ' * 50)
    followup=hit('b',text,'Compaction runbook')
    out=select_agent_context(req([initial,followup], 'compaction metrics counters disable'),2000,{initial.citation_id})
    parts=[e.chunk.text for e in out if e.chunk.doc_id=='b']
    assert any('budget_bytes_per_step' in s for s in parts)
    assert any('counters' in s for s in parts)
    assert sum(len(e.chunk.text) for e in out)<=2000

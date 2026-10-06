"""Independent review: saved scores only; no model inference, API calls, or experiments.
Run with OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ../envs/ml/bin/python scripts/review_cpu_checks.py.
"""
import json, re, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from persona_selection.mixture import em_weights, fit_and_evaluate, mixture_loglik, is_meta_response
ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(20261002)
def read(p): return [json.loads(l) for l in open(ROOT / p)]
def auc(p,n): return float(roc_auc_score([1]*len(p)+[0]*len(n),np.r_[p,n]))
def matched_auc(bp,bn,k,B=1200):
    # Marginal AUROC on question-balanced bags; same question sets in each class.
    q=np.array([rng.choice(len(bp),k,replace=False) for _ in range(B)])
    # Share question sets, as original analysis does; roc_auc_score compares all bags.
    def draw(b):
        width=max(map(len,b)); mat=np.array([np.pad(x,(0,width-len(x))) for x in b]); lengths=np.array(list(map(len,b)))
        pick=(rng.random(q.shape)*lengths[q]).astype(int)
        return mat[q,pick].sum(1)
    return auc(draw(bp),draw(bn))
def analyze(fn,pos,neg,a,b,extra_filter=False):
    rows=[r for r in read('results/subliminal/'+fn) if r['teacher'] in (pos,neg)]
    if extra_filter:
        pat=re.compile(r'\b(eagles?|talons?|raptors?|soar\w*|trains?|railways?|railroads?|locomotives?|rails?|stations?|tracks?|journeys?)\b',re.I)
        rows=[r for r in rows if not pat.search(r['completion'])]
    y=np.array([r['teacher']==pos for r in rows]); x=np.array([r['n_tokens'] for r in rows])
    s=np.array([r['ll']['k0:'+a]-r['ll']['k0:'+b] for r in rows]); s-=LinearRegression().fit(x[:,None],s).predict(x[:,None])
    by=defaultdict(lambda:[[],[]])
    for r,v,label in zip(rows,s,y): by[r['prompt']][0 if label else 1].append(float(v))
    pairs=[(np.array(p),np.array(n)) for p,n in by.values() if p and n]; bp,bn=map(list,zip(*pairs))
    delta=np.array([p.mean()-n.mean() for p,n in pairs]); within=np.array([auc(p,n) for p,n in pairs])
    ci=np.percentile(delta[rng.integers(len(delta),size=(2000,len(delta)))].mean(1),[2.5,97.5])
    null=[]
    for _ in range(999):
        ds=[]
        for p,n in pairs:
            z=rng.permutation(np.r_[p,n]); ds.append(z[:len(p)].mean()-z[len(p):].mean())
        null.append(np.mean(ds))
    out={'n_pos':int(y.sum()),'n_neg':int((~y).sum()),'n_common_questions':len(pairs),
         'raw_auc':auc(s[y],s[~y]),'within_question_auc_equal_weight':float(within.mean()),
         'equal_question_mean_gap_nats':float(delta.mean()),'question_boot_mean_gap_ci':ci.tolist(),
         'within_question_perm_p_one_sided':float((1+np.sum(np.array(null)>=delta.mean()))/1000),
         'token_means':[float(x[y].mean()),float(x[~y].mean())]}
    for k in [10,30]:
        out[f'matched_k{k}']=matched_auc(bp,bn,k,B=4000)
        vals=[]
        for _ in range(120):
            ix=rng.integers(len(pairs),size=len(pairs)); vals.append(matched_auc([bp[i] for i in ix],[bn[i] for i in ix],k))
        out[f'matched_k{k}_question_boot_ci']=np.percentile(vals,[2.5,97.5]).tolist()
    return out
out={}
for label,fn,pos,neg,a,b in [
 ('owl_eagle_text_base','scores_eagle20_olmo_base_text.jsonl','owl','eagle','owl','eagle'),
 ('owl_eagle_numbers_base','scores_eagle20_olmo_base_numbers.jsonl','owl','eagle','owl','eagle'),
 ('af_friend_text_base','scores_t11_base_text.jsonl','af','af_friend','af','af_friend'),
 ('student_af_friend_text_base','scores_stu16_base_text.jsonl','stu_af_text','stu_friend_text','af','af_friend')]:
    out[label]=analyze(fn,pos,neg,a,b); print(label,json.dumps(out[label]),flush=True)
out['owl_eagle_text_uniform_filter']=analyze('scores_eagle20_olmo_base_text.jsonl','owl','eagle','owl','eagle',True)
# Reproduce full-basis fits, both with and without the response filter.
for run in ['base_unknown_casual_v1','base_unknown_casual_short_v1','instruct_unknown_casual_v1']:
    d=ROOT/'results/phase1'/run; rows=read(str(d.relative_to(ROOT)/'rows.jsonl')); ll=[dict(r['ll']) for r in rows]
    for f in sorted(d.glob('rows_elicited*.jsonl')):
        for i,r in enumerate(read(str(f.relative_to(ROOT)))): ll[i].update(r['ll'])
    names=sorted(ll[0]); L=np.array([[r[n] for n in names] for r in ll]); l0=np.array([r['ll_generic'] for r in rows]); nt=np.array([r['n_tokens'] for r in rows]); groups=np.array([r['qidx'] for r in rows]); clean=np.array([not is_meta_response(r['response']) for r in rows])
    res={}
    for label,keep in [('clean',clean),('all',np.ones(len(rows),bool))]:
        fit=fit_and_evaluate(L[keep],l0[keep],groups[keep],nt[keep]); res[label]={'n':int(keep.sum()),'heldout':fit['heldout']}
    if run.startswith('instruct'):
        hold=set(json.loads((ROOT/'data/holdout_qids.json').read_text())); test=np.array([r['qid'] in hold for r in rows]); w,_=em_weights(L[~test]); mix=mixture_loglik(L[test],w)
        res['correct_reference_holdout']={'n':int(test.sum()),'mean_tokens_stored':float(nt[test].mean()),'mixture_stored_logp_per_token':float(mix.sum()/nt[test].sum()),'old_self_stored_per_token':float(sum(r['ll_self'] for r,t in zip(rows,test) if t)/nt[test].sum())}
        res['old_self_heldout']=fit_and_evaluate(L,np.array([r['ll_self'] for r in rows]),groups,nt)['heldout']
    out[run]=res; print(run,json.dumps(res),flush=True)
# Check stronger conclusions against inexpensive word/character baselines with held-out questions.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
for modality in ['text','numbers']:
    rows=read('results/subliminal/scores_eagle20_olmo_base_'+modality+'.jsonl'); rows=[r for r in rows if r['teacher'] in ('owl','eagle')]
    qs=np.array(sorted({r['prompt'] for r in rows})); prng=np.random.default_rng(123); prng.shuffle(qs); testq=set(qs[:len(qs)//2]); train=np.array([r['prompt'] not in testq for r in rows]); y=np.array([r['teacher']=='owl' for r in rows]); text=np.array([r['completion'] for r in rows]); res={}
    for kind,kw in [('word',{'ngram_range':(1,2),'min_df':2,'sublinear_tf':True}),('char',{'analyzer':'char','ngram_range':(2,4),'min_df':2,'sublinear_tf':True})]:
        pipe=make_pipeline(TfidfVectorizer(**kw),LogisticRegression(C=1,max_iter=1000)); pipe.fit(text[train],y[train]); score=pipe.decision_function(text[~train]); res[kind]={'heldout_auc':float(roc_auc_score(y[~train],score)),'train_n':int(train.sum()),'test_n':int((~train).sum())}
    out['owl_eagle_tfidf_'+modality]=res; print('tfidf',modality,res,flush=True)
(ROOT/'results/review_2026-10-02_cpu_checks.json').write_text(json.dumps(out,indent=2)+'\n')

"""Follow-up review: lightweight CPU analysis of saved results only, no model inference.
Run: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ../envs/ml/bin/python scripts/review_followup_cpu_checks.py
"""
import json,re,sys,runpy
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression
ROOT=Path(__file__).resolve().parents[1];rng=np.random.default_rng(20261006)
# Read the authoritative filter constant without importing generation/model libraries.
import ast
mod=ast.parse((ROOT/'scripts/subliminal_generate.py').read_text())
pattern=next(ast.literal_eval(n.value.args[0]) for n in mod.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='TEXT_FILTER' for t in n.targets))
FILTER=re.compile(pattern,re.I)
def read(fn):return [json.loads(l) for l in open(ROOT/fn)]
def auc(p,n):return float(roc_auc_score([1]*len(p)+[0]*len(n),np.r_[p,n]))
def bags(p,n,k=30,B=5000):
 nq=len(p);q=np.array([rng.choice(nq,k,replace=False) for _ in range(B)])
 def draw(vals):
  width=max(map(len,vals));mat=np.array([np.pad(x,(0,width-len(x))) for x in vals]);lens=np.array(list(map(len,vals)));pick=(rng.random(q.shape)*lens[q]).astype(int);return mat[q,pick].sum(1)
 return auc(draw(p),draw(n))
def features(r,extra):
 if not extra:return [r['n_tokens']]
 nums=np.array([int(x) for x in re.findall(r'\d+',r['completion'])]);s=r['completion']
 return [r['n_tokens'],len(nums),len(s),s.count(' '),s.count(','),float(nums.mean()),float(nums.std())]
def pair(fn,pos,neg,a,b,numbers=False,extra=False):
 rows=[r for r in read('results/subliminal/'+fn) if r['teacher'] in [pos,neg] and (numbers or not FILTER.search(r['completion']))]
 y=np.array([r['teacher']==pos for r in rows]);score=np.array([r['ll']['k0:'+a]-r['ll']['k0:'+b] for r in rows]);X=np.array([features(r,extra) for r in rows]);score-=LinearRegression().fit(X,score).predict(X)
 by=defaultdict(lambda:[[],[]])
 for r,v,z in zip(rows,score,y):by[r['prompt']][0 if z else 1].append(v)
 pairs=[(np.array(p),np.array(n)) for p,n in by.values() if p and n];p,n=map(list,zip(*pairs));delta=np.array([a.mean()-b.mean() for a,b in pairs])
 null=[]
 for _ in range(999):
  ds=[]
  for a,b in pairs:
   z=rng.permutation(np.r_[a,b]);ds.append(z[:len(a)].mean()-z[len(a):].mean())
  null.append(np.mean(ds))
 ci=np.percentile(delta[rng.integers(len(delta),size=(2000,len(delta)))].mean(1),[2.5,97.5])
 return {'n_q':len(p),'n_pos_common':sum(map(len,p)),'n_neg_common':sum(map(len,n)),
  'common_pool_k1':auc(np.concatenate(p),np.concatenate(n)),'matched_k10':bags(p,n,10),'matched_k30':bags(p,n),
  'equal_question_mean_gap':float(delta.mean()),'mean_gap_cluster_ci95':ci.tolist(),'within_q_permutation_p_one_sided':float((1+np.sum(np.array(null)>=delta.mean()))/1000),
  'within_q_auc':float(np.mean([auc(a,b) for a,b in pairs]))}
out={}
for numbers in [False,True]:
 for model in ['base','instruct']:
  fn='scores_nm_'+model+('_numbers.jsonl' if numbers else '_text.jsonl');pos,neg=('owl_nm_numbers','eagle_nm_numbers') if numbers else ('owl_nm','eagle_nm')
  for headers in [('owl_nm','eagle_nm'),('owl','eagle')]:
   key=f'nm_{model}_{"numbers" if numbers else "text"}_{headers[0]}';out[key]=pair(fn,pos,neg,*headers,numbers=numbers);print(key,json.dumps(out[key]),flush=True)
for key,fn,pos,neg,a,b in [('std_base_numbers','scores_eagle20_olmo_base_numbers.jsonl','owl','eagle','owl','eagle'),('nm_base_numbers','scores_nm_base_numbers.jsonl','owl_nm_numbers','eagle_nm_numbers','owl_nm','eagle_nm')]:
 out[key+'_format_adjusted']=pair(fn,pos,neg,a,b,numbers=True,extra=True);print(key+'_format_adjusted',out[key+'_format_adjusted'],flush=True)
# Paired uncertainty for mixture vs its best single component, fixed training fit.
ns=runpy.run_path(str(ROOT/'scripts/phase1_loglik_table.py'),run_name='review_import')
d=ns['load']('instruct_unknown_casual_v1');qs=np.unique(d['groups']);rr=np.random.default_rng(0);rr.shuffle(qs);test=np.isin(d['groups'],qs[:len(qs)//2]);w,_=ns['em_weights'](d['L'][~test]);mix=ns['mixture_loglik'](d['L'][test],w);L=d['L'][test];best=int(np.argmax(L.sum(0)));delta=mix-L[:,best];g=d['groups'][test];nt=d['nt'][test]
sums=np.array([[delta[g==q].sum(),nt[g==q].sum()] for q in np.unique(g)]);rep=sums[rng.integers(len(sums),size=(2000,len(sums)))].sum(1);rates=rep[:,0]/rep[:,1]
out['mixture_vs_single']={'best':d['names'][best],'paired_gain_per_token':float(delta.sum()/nt.sum()),'paired_bootstrap_se':float(rates.std()),'paired_ci95':np.percentile(rates,[2.5,97.5]).tolist(),'gain_per_response':float(delta.mean())};print('mixture_vs_single',out['mixture_vs_single'],flush=True)
# Multiway: corrected file, same original answer-split algorithm, no length adjustment, five seeds.
src=['control','hhh_teacher','af','af_friend','af_resent','af_owl','owl','trains'];hyp=['neutral','hhh','af','af_friend','af_resent','af_owl','owl','trains'];r=read('results/subliminal/scores_exact_instruct_text.jsonl');r=[x for x in r if not FILTER.search(x['completion'])];multi=[]
for seed in range(5):
 gen=np.random.default_rng(seed);acc=[]
 for source in src:
  M=np.array([[x['ll']['k0:'+h] for h in hyp] for x in r if x['teacher']==source]);ix=gen.permutation(len(M));ev=M[ix[len(M)//2:]]
  acc.append([float(np.mean(ev[gen.integers(len(ev),size=(2000,k))].sum(1).argmax(1)==src.index(source))) for k in [1,5,10,30]])
 multi.append(np.mean(acc,0).tolist())
out['exact_multiway_original_answer_split_five_seeds']=multi;print('multiway',multi,flush=True)
# Metadata and factual provenance checks.
for teacher in ['owl','eagle','control','owl_nm','eagle_nm','trains_nm','owl_nm_numbers','eagle_nm_numbers','trains_nm_numbers']:
 m=json.load(open(ROOT/'results/subliminal'/teacher/'meta.json'));jpath=ROOT/'results/subliminal'/teacher/'judge.json';out['meta_'+teacher]={k:m[k] for k in ['text_raw','text_kept','text_filtered','numbers_raw','numbers_kept']};out['meta_'+teacher]['fav_rate']=m['favorite_animal']['rate'];
 if jpath.exists():out['meta_'+teacher]['judge']=json.load(open(jpath))['summary']
(ROOT/'results/review_2026-10-06_cpu_checks.json').write_text(json.dumps(out,indent=2)+'\n')

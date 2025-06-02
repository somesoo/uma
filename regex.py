"""
optimize_kmer_regex.py — recall‑heavy optimisation of discriminative k‑mer motifs.

*Works as an in‑place upgrade of the previous version (same name).*

-----------------------------------
Key new features
-----------------------------------
1. **Grid search (`--grid`)** across threshold pairs `(min_pos, max_neg)` and multiple *k* values.
2. **F‑beta optimisation** (`--beta`, default **2.0**) to favour recall over precision.
3. **Target recall** (`--target_recall`) — automatically relax thresholds until this recall is achieved.
4. Clean terminal report (**precision, recall, F‑beta**) + option to write motifs and combined regex.

-----------------------------------
Example: maximise recall with reasonable FP
-----------------------------------
python optimize_kmer_regex.py --pos class1.txt --neg class0.txt \
      --k 6 7 --grid --beta 2.0 --max_neg_grid 10 --output motifs.txt

Example: guarantee ≥ 0.85 recall (allowing many FP)
-----------------------------------
python optimize_kmer_regex.py --pos class1.txt --neg class0.txt \
      --target_recall 0.85 --max_neg 20
"""

import argparse
import collections
import itertools
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

# ------------------ IUPAC helpers ------------------ #
IUPAC_MAP = {
    frozenset({'A'}): 'A', frozenset({'C'}): 'C', frozenset({'G'}): 'G', frozenset({'T'}): 'T',
    frozenset({'A','G'}): 'R', frozenset({'C','T'}): 'Y', frozenset({'G','C'}): 'S', frozenset({'A','T'}): 'W',
    frozenset({'G','T'}): 'K', frozenset({'A','C'}): 'M',
    frozenset({'C','G','T'}): 'B', frozenset({'A','G','T'}): 'D',
    frozenset({'A','C','T'}): 'H', frozenset({'A','C','G'}): 'V',
    frozenset({'A','C','G','T'}): 'N'
}

def chars2code(chars: Set[str], use_iupac: bool) -> str:
    if len(chars) == 1:
        return next(iter(chars))
    key = frozenset(chars)
    if use_iupac:
        return IUPAC_MAP.get(key, 'N')  # fallback na N jeśli brak odwzorowania
    else:
        return "[" + "".join(sorted(chars)) + "]"

# ------------------ IO ------------------ #

def read_sequences(path:str)->List[str]:
    seqs=[]
    with open(path) as fh:
        for line in fh:
            line=line.strip()
            if not line or line.startswith('>'): continue
            seqs.append(re.sub(r'[^ACGTacgt]','',line).upper())
    return seqs

# ------------------ k‑mer utils ------------------ #

def sliding_kmers(seq:str,k:int):
    for i in range(len(seq)-k+1):
        yield seq[i:i+k], i

def count_kmers(seqs:Sequence[str],k:int)->Dict[str,int]:
    c=collections.Counter()
    for s in seqs:
        for km,_ in sliding_kmers(s,k): c[km]+=1
    return c

def filter_kmers(pos_counts,neg_counts,min_pos,max_neg):
    return [k for k,c in pos_counts.items() if c>=min_pos and neg_counts.get(k,0)<=max_neg]

def extend_motif(kmer:str, seqs_pos:Sequence[str], flank:int, use_iupac:bool)->str:
    k=len(kmer)
    left_sets=[set() for _ in range(flank)]
    right_sets=[set() for _ in range(flank)]
    for seq in seqs_pos:
        for i in range(len(seq)-k+1):
            if seq[i:i+k]!=kmer: continue
            for f in range(1,flank+1):
                if i-f>=0: left_sets[flank-f].add(seq[i-f])
                if i+k-1+f<len(seq): right_sets[f-1].add(seq[i+k+f-1])
    left=''.join(chars2code(s,use_iupac) for s in left_sets)
    right=''.join(chars2code(s,use_iupac) for s in right_sets)
    return f'{left}{kmer}{right}'

def build_regex(motifs:Sequence[str])->str:
    return '('+'|'.join(motifs)+')'

def evaluate(seqs:Sequence[str], pattern:re.Pattern)->Tuple[int,int]:
    matched=sum(bool(pattern.search(s)) for s in seqs)
    return matched, len(seqs)-matched

def fbeta(tp,fp,fn,beta:float):
    if tp==0: return 0.0
    prec=tp/(tp+fp)
    rec=tp/(tp+fn)
    b2=beta*beta
    return (1+b2)*prec*rec/(b2*prec+rec)

# ------------------ optimisation engines ------------------ #

def run_single(seqs_pos,seqs_neg,k,min_pos,max_neg,flank,iupac):
    pos_counts=count_kmers(seqs_pos,k)
    neg_counts=count_kmers(seqs_neg,k)
    kmers=filter_kmers(pos_counts,neg_counts,min_pos,max_neg)
    if not kmers: return None
    motifs=[extend_motif(km,seqs_pos,flank,iupac) for km in kmers]
    regex=build_regex(motifs)
    pat=re.compile(regex)
    tp,fn=evaluate(seqs_pos,pat)
    fp,tn=evaluate(seqs_neg,pat)
    return {'k':k,'min_pos':min_pos,'max_neg':max_neg,'motifs':motifs,'regex':regex,'tp':tp,'fp':fp,'fn':fn}

def grid_search(seqs_pos,seqs_neg,ks,min_pos_grid,max_neg_grid,flank,iupac,beta):
    best=None
    for k in ks:
        for min_pos in range(min_pos_grid,0,-1):
            for max_neg in range(0,max_neg_grid+1):
                res=run_single(seqs_pos,seqs_neg,k,min_pos,max_neg,flank,iupac)
                if not res: continue
                score=fbeta(res['tp'],res['fp'],res['fn'],beta)
                if best is None or score>best['score']:
                    res['score']=score
                    best=res
    return best

def reach_target(seqs_pos,seqs_neg,k,flank,iupac,target_recall,max_neg):
    for min_pos in range(6,0,-1):
        res=run_single(seqs_pos,seqs_neg,k,min_pos,max_neg,flank,iupac)
        if res and res['tp']/len(seqs_pos) >= target_recall:
            res['score']=res['tp']/len(seqs_pos)
            return res
    return None

# ------------------ CLI ------------------ #

def main():
    p=argparse.ArgumentParser(description='Recall‑oriented k‑mer regex optimiser')
    p.add_argument('--pos', required=True)
    p.add_argument('--neg', required=True)
    p.add_argument('--k', type=int, nargs='+', default=[6])
    p.add_argument('--flank', type=int, default=1)
    p.add_argument('--iupac', action='store_true')
    p.add_argument('--grid', action='store_true', help='grid search mode')
    p.add_argument('--min_pos', type=int, default=4)
    p.add_argument('--max_neg', type=int, default=1)
    p.add_argument('--max_pos_grid', type=int, default=6)
    p.add_argument('--max_neg_grid', type=int, default=5)
    p.add_argument('--beta', type=float, default=2.0, help='F‑beta weight (β>1 favours recall)')
    p.add_argument('--target_recall', type=float, help='required recall (bypass grid)')
    p.add_argument('--output', help='write motifs + regex to file')
    args=p.parse_args()

    seqs_pos=read_sequences(args.pos)
    seqs_neg=read_sequences(args.neg)

    if args.target_recall:
        best=None
        for k in args.k:
            res=reach_target(seqs_pos,seqs_neg,k,args.flank,args.iupac,args.target_recall,args.max_neg)
            if res: best=res if best is None or res['tp']>best['tp'] else best
        if not best:
            sys.exit('Could not reach target recall with given parameters.')
        best['score']=best['tp']/len(seqs_pos)
    elif args.grid:
        best=grid_search(seqs_pos,seqs_neg,args.k,args.max_pos_grid,args.max_neg_grid,args.flank,args.iupac,args.beta)
        if not best:
            sys.exit('Grid search failed to find motifs with current limits.')
    else:
        best=run_single(seqs_pos,seqs_neg,args.k[0],args.min_pos,args.max_neg,args.flank,args.iupac)
        if not best:
            sys.exit('No motifs found with provided thresholds.')

    tp,fp,fn=best['tp'],best['fp'],best['fn']
    precision=tp/(tp+fp) if tp+fp else 0
    recall=tp/(tp)
               
if __name__ == "__main__":
    main()

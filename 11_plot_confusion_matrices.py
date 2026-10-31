"""Step 11 – Draw the confusion-matrix figures of Chapter 4

The script reads the saved predictions and draws three figures: UC1 matrices (Figure 2), UC2 matrices (Figure 3)
and the UC2 share flagged as phishing per scenario and technique (Figure 4).
The GPT-4o UC1 counts are rebuilt from the saved metrics of the 200/200 sample.

Thesis: Chapter, Section 4.1 (Figure 2) and Section 4.2 (Figures 3–4).

Reads:
READ_DIR/uc1_predictions.csv — UC1 predictions from 03_run_uc1_baselines.py.
READ_DIR/bert_predictions.csv — BERT predictions from 05_bert_uc1_colab.ipynb.
READ_DIR/uc2_predictions.csv — UC2 predictions from 08_run_uc2_evaluation.py.
DATA_DIR/uc2_dataset.csv — UC2 dataset from 06, for the lure-domain age.

Writes: SAVE_DIR/cm_uc1.png, cm_uc2.png, cm_uc2_scenario.png — Figures 2, 3 and 4.

Usage: python 11_plot_confusion_matrices.py
"""
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

# Folders used by this script (defaults from config.py; change here or in config.py).
READ_DIR = C.RESULTS_DIR                        # prediction files (from 03, 05 and 08)
DATA_DIR = C.PROCESSED_DIR                      # uc2_dataset.csv (from 06)
SAVE_DIR = C.RESULTS_DIR / 'confusion_matrices' # figures
os.makedirs(SAVE_DIR, exist_ok=True)

# --- Figure style ---
plt.rcParams.update({'font.family':'serif',
                     'font.serif':['Liberation Serif','Times New Roman','DejaVu Serif'],
                     'font.size':9})
cmap=LinearSegmentedColormap.from_list('b',['#ffffff','#cde2fb','#86b6ef','#3987e5','#1c5cab',
                                            '#0d366b'])
INK='#1f1f1f'   # near-black color for text and lines


def panel(ax,tp,fp,fn,tn,title,n):
    """Draw one 2x2 confusion matrix; color = percentage of the actual class."""
    M=np.array([[tn,fp],[fn,tp]],float); R=M/M.sum(1,keepdims=True)
    ax.imshow(R,cmap=cmap,vmin=0,vmax=1)
    for i in range(2):
        for j in range(2):
            lab=[['TN','FP'],['FN','TP']][i][j]
            ax.text(j,i,f'{lab}\n{int(M[i,j]):,}\n{R[i,j]*100:.1f}%',ha='center',va='center',
                    color='white' if R[i,j]>0.55 else INK,fontsize=8.5,linespacing=1.25)
    ax.set_xticks([0,1],['Legitimate','Phishing']); ax.set_yticks([0,1],['Legitimate','Phishing'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title(f'{title}\n(n = {n:,})',fontsize=9)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_xticks([.5],minor=True); ax.set_yticks([.5],minor=True)
    ax.grid(which='minor',color='white',linewidth=2); ax.tick_params(which='both',length=0)
    ax.set_xlim(-.5,1.5); ax.set_ylim(1.5,-.5)


def grid(panels,cols,fn):
    """Arrange several confusion matrices in a grid and save the figure."""
    rows=int(np.ceil(len(panels)/cols))
    fig,axs=plt.subplots(rows,cols,figsize=(2.35*cols,2.55*rows))
    axs=np.atleast_1d(axs).ravel()
    for ax,p in zip(axs,panels): panel(ax,*p)
    for ax in axs[len(panels):]: ax.axis('off')
    fig.tight_layout(); fig.savefig(fn,dpi=220,bbox_inches='tight'); plt.close(fig)


def cnt(y,p):
    """Return (TP, FP, FN, TN) for labels y and predictions p."""
    y=np.asarray(y);p=np.asarray(p)
    return ((y==1)&(p==1)).sum(),((y==0)&(p==1)).sum(),((y==1)&(p==0)).sum(),((y==0)&(p==0)).sum()

# --- 1. Figure 2: UC1 confusion matrices ---
u=pd.read_csv(READ_DIR / 'uc1_predictions.csv'); b=pd.read_csv(READ_DIR / 'bert_predictions.csv')
P=[]
for col,name in [('tfidf_lr_pred','TF-IDF + LR'),('tfidf_rf_pred','TF-IDF + RF'),
                 ('tfidf_lr_meta_pred','TF-IDF + metadata (LR)')]:
    P.append((*cnt(u.label,u[col]),name,len(u)))
P.append((*cnt(b.label,b.bert_pred),'BERT-base (fine-tuned)',len(b)))
# GPT-4o UC1: counts derived from the saved metrics on the 200/200 sample.
P.append((185,3,15,197,'GPT-4o few-shot',400))
P.append((171,2,29,198,'GPT-4o zero-shot / CoT',400))
grid(P,3,os.path.join(SAVE_DIR,'cm_uc1.png')); uc1=P

# --- 2. Figure 3: UC2 confusion matrices ---
# Non-LLM approaches use all 3,000 emails; GPT-4o uses the 750-email subsample.
p=pd.read_csv(READ_DIR / 'uc2_predictions.csv'); s=p[p.in_llm_sample==True]
Q=[]
for col,name in [('reputation_baseline_pred','Reputation baseline'),
                 ('first_contact_heuristic_pred','First-contact heuristic'),
                 ('uc1_tfidf_lr_meta_pred','UC1 tfidf_lr_meta (transferred)')]:
    Q.append((*cnt(p.label,p[col]),name,len(p)))
for col,name in [('llm_few_shot_pred','GPT-4o few-shot (baseline)'),
                 ('llm_chain_of_thought_pred','GPT-4o CoT (baseline)'),
                 ('llm_few_shot_improved_prompt_pred','GPT-4o few-shot (improved)'),
                 ('llm_chain_of_thought_improved_prompt_pred','GPT-4o CoT (improved)')]:
    Q.append((*cnt(s.label,s[col]),name,len(s)))
grid(Q,4,os.path.join(SAVE_DIR,'cm_uc2.png'))
for q in uc1+Q: print(q)

# --- 3. Figure 4: share flagged as phishing, per scenario and technique ---
d=pd.read_csv(DATA_DIR / 'uc2_dataset.csv'); p['lure']=d.lure_domain_aged
def grp(r):
    """Return the Figure 4 row label (scenario, technique and lure age) for one email."""
    if r.scenario!='trusted_source_phishing':
        return {'legitimate':'Legitimate','overt_phishing':'Overt phishing'}[r.scenario]
    if r.technique=='typosquat_link': return 'Trusted-source: typosquat-link'
    return ('Trusted-source: device-code, aged lure' if r.lure==True
            else 'Trusted-source: device-code, fresh lure')
p['g']=p.apply(grp,axis=1)
rows=['Legitimate','Overt phishing','Trusted-source: typosquat-link',
      'Trusted-source: device-code, aged lure','Trusted-source: device-code, fresh lure']
# (prediction column, figure label, is GPT-4o); GPT-4o is scored on the subsample only.
cols=[('reputation_baseline_pred','Reputation\nbaseline',False),
      ('first_contact_heuristic_pred','First-contact\nheuristic',False),
      ('uc1_tfidf_lr_meta_pred','UC1 classifier\n(transferred)',False),
      ('llm_few_shot_pred','GPT-4o FS\n(baseline)',True),
      ('llm_chain_of_thought_pred','GPT-4o CoT\n(baseline)',True),
      ('llm_few_shot_improved_prompt_pred','GPT-4o FS\n(improved)',True),
      ('llm_chain_of_thought_improved_prompt_pred','GPT-4o CoT\n(improved)',True)]
V=np.zeros((len(rows),len(cols))); N=np.zeros_like(V,dtype=int)
for j,(c,_,llm) in enumerate(cols):
    src=p[p.in_llm_sample==True] if llm else p
    for i,r in enumerate(rows):
        g=src[src.g==r]; V[i,j]=g[c].mean(); N[i,j]=len(g)
fig,ax=plt.subplots(figsize=(8.6,3.4))
ax.imshow(V,cmap=cmap,vmin=0,vmax=1,aspect='auto')
for i in range(len(rows)):
    for j in range(len(cols)):
        ax.text(j,i,f'{V[i,j]*100:.1f}%\nn={N[i,j]}',ha='center',va='center',fontsize=8,
                color='white' if V[i,j]>0.55 else INK)
ax.set_xticks(range(len(cols)),[c[1] for c in cols],fontsize=8)
ax.set_yticks(range(len(rows)),rows,fontsize=8.5)
ax.xaxis.tick_top()
ax.set_xticks(np.arange(len(cols)+1)-.5,minor=True)
ax.set_yticks(np.arange(len(rows)+1)-.5,minor=True)
ax.grid(which='minor',color='white',linewidth=2)
for sp in ax.spines.values(): sp.set_visible(False)
ax.tick_params(which='both',length=0)
ax.axhline(0.5,color=INK,linewidth=1.2)
fig.tight_layout()
fig.savefig(os.path.join(SAVE_DIR,'cm_uc2_scenario.png'),dpi=220,bbox_inches='tight')
print(pd.DataFrame(V,index=rows,columns=[c[0] for c in cols]).round(3)); print(N)

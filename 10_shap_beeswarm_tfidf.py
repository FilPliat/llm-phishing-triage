"""Step 10 – Draw SHAP beeswarm plots for the TF-IDF + metadata model

The script retrains tfidf_lr_meta as in 03 and computes exact SHAP values with LinearExplainer.
It draws one beeswarm plot for UC1 (2,000 test emails) and one for UC2 (all 3,000 emails, no retraining).
Main finding: the number of URLs dominates, so on UC2 the model flags almost every email with a link.
Needs the shap package.

Thesis: Chapter 4, Section 4.3 (Figures 10–11) and Chapter 5.

Reads:
READ_DIR/train.csv, test.csv — UC1 splits from 02_prepare_data.py.
READ_DIR/uc2_dataset.csv — UC2 dataset from 06_build_uc2_dataset.py.

Writes:
SAVE_DIR/beeswarm_uc1.png — UC1 plot (Figure 10).
SAVE_DIR/beeswarm_uc2.png — UC2 plot (Figure 11).

Usage: python 10_shap_beeswarm_tfidf.py
"""
import sys, os, numpy as np, pandas as pd, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from src import baselines as B

# Folders used by this script (defaults from config.py; change here or in config.py).
READ_DIR = C.PROCESSED_DIR                  # train.csv, test.csv (02) and uc2_dataset.csv (06)
SAVE_DIR = C.RESULTS_DIR / "shap_beeswarm"  # beeswarm plots
os.makedirs(SAVE_DIR, exist_ok=True)


# --- 1. Load the data ---
def load(p):
    """Read a dataset CSV and clean the three columns the model uses."""
    df=pd.read_csv(p)
    for c in ('text','sender_tld'): df[c]=df[c].fillna('').astype(str)
    df['num_urls']=pd.to_numeric(df['num_urls'],errors='coerce').fillna(0); return df
tr=load(READ_DIR / 'train.csv'); te=load(READ_DIR / 'test.csv')
u2=load(READ_DIR / 'uc2_dataset.csv')

# --- 2. Retrain the TF-IDF + metadata model as in 03 ---
import shap, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
M=B.MODELS['tfidf_lr_meta'][0](); cols=['text','sender_tld','num_urls']
M.fit(tr[cols],tr.label.to_numpy(int))
pre,clf=M.named_steps['pre'],M.named_steps['clf']
# Readable feature names for the plots.
names=[n.split('__',1)[1] for n in pre.get_feature_names_out()]
names=[('sender TLD = '+(n[len('sender_tld_'):] or '(none)').replace('infrequent_sklearn','other'))
       if n.startswith('sender_tld_') else ('number of URLs' if n=='num_urls' else f'"{n}"')
       for n in names]

# --- 3. Exact SHAP values (background = full training set) ---
Xtr=pre.transform(tr[cols]);
bg=Xtr
ex=shap.LinearExplainer(clf,bg)
def sv(X):
    """Return raw SHAP values for X (unused)."""
    v=ex.shap_values(X); return v

te_s=te.sample(2000,random_state=42)
X1=pre.transform(te_s[cols]); S1=ex(X1); S1.feature_names=names
p2=M.predict(u2[cols]); u2['pred']=p2
leg=u2[u2.scenario=='legitimate']
X2=pre.transform(leg[cols]); S2=ex(X2); S2.feature_names=names
X3=pre.transform(u2[cols]); S3=ex(X3); S3.feature_names=names
print('UC2 legit FP rate',leg.pred.mean(), 'n',len(leg))

# --- 4. Print the top features by mean |SHAP| ---
def top(S,k=12,mask=None):
    """Return the k features with the largest mean |SHAP| as (name, mean |SHAP|, mean SHAP)."""
    v=S.values if mask is None else S.values[mask]
    m=np.abs(v).mean(0); idx=np.argsort(-m)[:k]
    return [(names[i],round(m[i],4),round(v[:,i].mean(),4)) for i in idx]
print('UC1 top',top(S1)); print('UC2 legit top',top(S2))
fp=leg.pred.values==1
print('UC2 legit FP only top',top(S2,mask=fp)); print('UC2 legit TN only top',top(S2,mask=~fp))

# --- 5. Draw the two beeswarm plots (Figures 10 and 11) ---
plt.rcParams.update({'font.family':'serif','font.serif':['Liberation Serif','DejaVu Serif'],
                     'font.size':10})
for S,fn,title in [(S1,'beeswarm_uc1.png','UC1 test set (random 2,000 of 13,030 emails)'),
                   (S3,'beeswarm_uc2.png',
                    'UC2 pool (all 3,000 emails), classifier transferred without retraining')]:
    plt.figure()
    shap.plots.beeswarm(S,max_display=15,show=False,plot_size=(8,6.2),
                        color_bar_label='Feature value (TF-IDF weight / count)')
    ax=plt.gca()
    if fn=='beeswarm_uc1.png':
        # Clip the few emails with very many URLs and note them on the plot.
        k=int((S.values[:,names.index('number of URLs')]>10).sum())
        others=S.values.sum(1)-S.values[:,np.argsort(-np.abs(S.values).mean(0))[:14]].sum(1)
        ax.set_xlim(min(-5,others.min()-0.5),10)
        ax.annotate(f'off-scale: {k} emails\n(up to +{S.values.max():.0f})',
                    xy=(9.8,ax.get_ylim()[1]-1.5),ha='right',va='center',fontsize=8.5,
                    color='#444')
    ax.set_xlabel('SHAP value (impact on log-odds of "phishing")'); ax.set_title(title,fontsize=10)
    plt.tight_layout(); plt.savefig(os.path.join(SAVE_DIR,fn),dpi=220,bbox_inches='tight')
    plt.close('all')

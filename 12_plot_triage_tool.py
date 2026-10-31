"""Step 12 – Draw the layer diagram of the triage tool

The script draws the six layers of the GPT-4o triage tool, the comparison models and the shared evaluation.
It only draws boxes and arrows.

Thesis: Chapter 3, Section 3.2 (Figure 1).

Writes: SAVE_DIR/triage_tool.png — the triage tool diagram (Figure 1).

Usage: python 12_plot_triage_tool.py
"""
import os
import sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

# Folders used by this script (defaults from config.py; change here or in config.py).
SAVE_DIR = C.RESULTS_DIR        # triage_tool.png

# --- Figure style ---
plt.rcParams.update({'font.family':'serif','font.serif':['Liberation Serif','DejaVu Serif']})
INK='#1f1f1f'   # near-black for text, outlines and arrows
CORE='#1c5cab'  # dark blue fill for the GPT-4o inference layer
PL='#cde2fb'    # light blue fill for the other tool layers
GR='#f2f2f2'    # light gray fill for the comparison and evaluation boxes
# Canvas: x from 0 to 100, y from 0 to 64.
fig,ax=plt.subplots(figsize=(10,6.4)); ax.set_xlim(0,100); ax.set_ylim(0,64); ax.axis('off')
def box(x,y,w,h,title,lines=(),fill=PL,tc=INK,num=None,dashed=False):
    """Draw a rounded box with a bold title and a few lines of text."""
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.4,rounding_size=1.2',fc=fill,ec=INK,lw=1.1,ls=(0,(4,3)) if dashed else '-'))
    t=f'{num}. {title}' if num else title
    ax.text(x+w/2,y+h-1.8,t,ha='center',va='top',fontsize=9.5,weight='bold',color=tc)
    for i,l in enumerate(lines): ax.text(x+w/2,y+h-5.2-i*2.5,l,ha='center',va='top',fontsize=8,color=tc)
def arrow(x1,y1,x2,y2,both=False,label=None,lx=0,ly=0):
    """Draw an arrow from (x1, y1) to (x2, y2), with an optional label."""
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='<|-|>' if both else '-|>',mutation_scale=11,lw=1.1,color=INK))
    if label: ax.text((x1+x2)/2+lx,(y1+y2)/2+ly,label,ha='center',va='center',fontsize=8,style='italic',color='#333')
# --- Title and frame ---
ax.text(50,63.2,'Triage tool (implemented in this thesis)',ha='center',va='top',fontsize=10.5,weight='bold')
ax.add_patch(FancyBboxPatch((0.2,20.5),99.6,40,boxstyle='round,pad=0.3,rounding_size=1.5',fc='none',ec='#888',lw=0.9,ls=(0,(2,2))))
# --- Top row: layers 1-5 ---
box(1,40,17,17,'Data layer',['UC1 corpora / UC2','dataset → record:','subject, body,','sender, domain,','TLD, URLs'],num=1)
box(21,40,17,17,'Enrichment',['(UC2 only)','domain age','DGA score','VirusTotal-style','detections'],num=2)
box(41,40,18,17,'Prompt layer',['system prompt','(SOC analyst, JSON)','zero-shot / few-shot /','chain-of-thought','+ tenant check (UC2)'],num=3)
box(62,40,15,17,'Inference',['GPT-4o','gpt-4o-2024-08-06','temperature 0','JSON mode'],fill=CORE,tc='white',num=4)
box(80,40,19,17,'Output layer',['triage object:','classification,','confidence, severity,','IOCs, action,','rationale'],num=5)
for x1,x2 in [(18.8,20.2),(38.8,40.2),(59.8,61.2),(77.8,79.2)]: arrow(x1,48.5,x2,48.5)
# --- Logging layer ---
box(58,23,23,10,'Logging layer',['response cache, tokens,','cost, latency'],num=6)
arrow(69.5,39.3,69.5,33.8,both=True)
# --- Bottom row: comparison models and evaluation ---
box(1,3,40,12.5,'Comparison models',['TF-IDF + Logistic Regression / Random Forest','TF-IDF + metadata, fine-tuned BERT','reputation rule, first-contact heuristic'],fill=GR)
box(50,3,49,12.5,'Evaluation (same code and samples for all)',['confusion matrix, precision, recall, F1,','accuracy, false-positive rate, ROC-AUC','explainability: attention (BERT), SHAP'],fill=GR)
arrow(41.8,9.2,49.2,9.2)
arrow(89.5,39.2,89.5,16.3,label='verdicts',lx=4.5)
# --- Save the figure ---
os.makedirs(SAVE_DIR, exist_ok=True)
plt.savefig(os.path.join(SAVE_DIR, 'triage_tool.png'), dpi=220, bbox_inches='tight')
print('Saved ->', os.path.join(SAVE_DIR, 'triage_tool.png'))

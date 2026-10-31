"""Step 13 – Draw the proposed modular triage architecture

The script draws the proposed architecture, from the input adapter to the output sinks and future analysis tools.
Solid boxes are implemented in this thesis; dashed boxes are proposed extensions.

Thesis: Chapter 5, Section 5.3 (Figure 12).

Writes: SAVE_DIR/architecture.png — the architecture diagram (Figure 12).

Usage: python 13_plot_architecture.py
"""
import os
import sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

# Folders used by this script (defaults from config.py; change here or in config.py).
SAVE_DIR = C.RESULTS_DIR        # architecture.png

# --- Figure style ---
plt.rcParams.update({'font.family':'serif','font.serif':['Liberation Serif','DejaVu Serif']})
INK='#1f1f1f'   # near-black for text, outlines and arrows
CORE='#1c5cab'  # dark blue fill for the LLM core
PL='#cde2fb'    # light blue fill for the implemented components
FUT='#ffffff'   # white fill for proposed (future) components
RED='#D4003A'   # red outline of the future-work analysis tools
# Canvas: x from 0 to 100, y from 0 to 58.5.
fig,ax=plt.subplots(figsize=(10,5.6)); ax.set_xlim(0,100); ax.set_ylim(0,58.5); ax.axis('off')
def box(x,y,w,h,title,lines=(),fill=PL,dashed=False,tc=INK,edge=INK):
    """Draw a rounded box with a bold title; dashed = proposed component."""
    p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.4,rounding_size=1.2',fc=fill,ec=edge,lw=1.1,ls=(0,(4,3)) if dashed else '-')
    ax.add_patch(p)
    ax.text(x+w/2,y+h-2.2,title,ha='center',va='top',fontsize=9.5,weight='bold',color=tc)
    for i,l in enumerate(lines): ax.text(x+w/2,y+h-5.6-i*2.6,l,ha='center',va='top',fontsize=8,color=tc)
def arrow(x1,y1,x2,y2,dashed=False,both=False,label=None,lx=0,ly=0):
    """Draw an arrow from (x1, y1) to (x2, y2); dashed = proposed connection."""
    a=FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='<|-|>' if both else '-|>',mutation_scale=11,lw=1.1,color=INK,ls=(0,(4,3)) if dashed else '-')
    ax.add_patch(a)
    if label: ax.text((x1+x2)/2+lx,(y1+y2)/2+ly,label,ha='center',va='center',fontsize=8,style='italic',color='#333')
# --- Top row: the implemented pipeline ---
box(1,30,13,15,'Input adapter',['reported-phish','mailbox, .eml,','SIEM alert'])
box(18,30,16,15,'Enrichment plugins',['domain age','DGA score','reputation hits','prior contact'])
box(38,30,17,15,'Prompt builder',['+ prompt modules,','e.g. the tenant-','identity check','(Section 3.3)'])
box(59,30,15,15,'LLM core',['orchestrator','(GPT-4o in this','thesis)'],fill=CORE,tc='white')
box(78,30,21,15,'Triage object (JSON)',['verdict, confidence,','severity, IOCs,','recommended action,','rationale'])
for x1,x2 in [(14.8,17.2),(34.8,37.2),(55.8,58.2),(74.8,77.2)]: arrow(x1,37.5,x2,37.5)
# --- Optional pre-filter above the LLM core ---
box(42,49.5,32,7,'Pre-filter plugin (optional)',['classical classifier as a cheap first pass'],fill=PL,dashed=True)
arrow(66.5,48.8,66.5,45.9)
# --- Output sinks under the triage object ---
box(78,12,21,12,'Output sinks',['SOAR / case ticket','analyst UI'],fill=FUT,dashed=True)
arrow(88.5,29.2,88.5,24.8)
# --- Future-work analysis tools ---
box(22,1,48,21,'Analysis tools (future work, Section 5.4)',[],fill=FUT,dashed=True,edge=RED)
tools=[('URL','detonation, redirect chain'),('Page render','screenshot + brand check'),('Attachment','sandbox'),('QR / OCR','decoder')]
for i,(a,b) in enumerate(tools):
    x=24+i*11.3; box(x,4,9.6,10.5,a,[x.strip() for x in b.split(',')] if ',' in b else ([b] if len(b)<14 else [b.split(' ',1)[0],b.split(' ',1)[1]]),fill='#f6f6f6')
arrow(66.5,29.2,66.5,22.8,dashed=True,both=True,label='calls tools /\nreceives evidence table',lx=9,ly=0)
# --- Legend ---
ax.plot([74,78],[6,6],color=INK,lw=1.1); ax.text(79,6,'implemented in this thesis',va='center',fontsize=8)
ax.plot([74,78],[2.5,2.5],color=INK,lw=1.1,ls=(0,(4,3))); ax.text(79,2.5,'proposed / future work',va='center',fontsize=8)
# --- Save the figure ---
os.makedirs(SAVE_DIR, exist_ok=True)
plt.savefig(os.path.join(SAVE_DIR, 'architecture.png'),dpi=220,bbox_inches='tight')

"""Regenerate four supplementary figures from frozen supplied evidence; no fitting."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
E=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=E/'regenerated_figures')
OUT=p.parse_args().output.resolve();OUT.mkdir(parents=True,exist_ok=True)
EXTERNAL=E/'trajectory_arrays'
DATASETS=['XJTU','MATR','Tongji']
def figures(stages, domain, calce):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.bbox': 'tight', 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.5), constrained_layout=True)
    for ax, ds in zip(axes, DATASETS):
        s = stages[stages.dataset==ds]
        ax.plot(s.single_query_ms, s.mae_pp, 'o-', color='#236491')
        for _, row in s.iterrows():
            ax.annotate(row.group, (row.single_query_ms,row.mae_pp), xytext=(3,6), textcoords='offset points', fontsize=9)
        ax.set_title(ds); ax.set_xlabel('Single-query inference (ms)'); ax.set_ylabel('MAE (SOH pp)'); ax.grid(alpha=.18)
        ax.margins(x=.14, y=.22)
    fig.suptitle('Existing frozen stages: accuracy and measured inference cost', fontsize=13)
    for ext in ['png','pdf']: fig.savefig(OUT / ('stage_cost.'+ext), dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9,3.8), constrained_layout=True)
    labels = {'CS2_low_SOC_RPT_0.22A':'CS2 low-SOC RPT (n=2)', 'CX2_alternating_pulse_full_reference':'CX2 alternating pulse (n=1)',
              'PL_full_SOC_0.5C':'PL full-SOC 0.5C discharge (n=2)', 'PL_full_SOC_2C':'PL full-SOC 2C discharge (n=2)'}
    vals = calce.delta_full_minus_B_pp
    bars = ax.barh([labels[d] for d in calce.domain], vals, color=['#bd573b' if v>0 else '#287d7d' for v in vals])
    for bar, v in zip(bars, vals):
        ax.text(v+(.025 if v>=0 else -.025),bar.get_y()+bar.get_height()/2, f'{v:+.3f}', ha='left' if v>=0 else 'right', va='center')
    ax.axvline(0,color='#777',lw=.8); ax.set_xlim(-.95,2.35); ax.invert_yaxis()
    ax.set_xlabel('MAE change: B1234 minus B (SOH pp); negative favors B1234')
    ax.set_title(f'CALCE: equal-protocol mean change = {vals.mean():+.3f} pp\nEach protocol result averages the same six frozen source models')
    for ext in ['png','pdf']: fig.savefig(OUT / ('calce_protocols.'+ext), dpi=180)
    plt.close(fig)

    pivot = domain.pivot(index=['dataset','domain'], columns='group', values='mae').reindex(DATASETS, level=0)
    columns = [('B','B1','M1 added to B'), ('B1','B12','M2 added to B1'), ('B12','B123','M3 added to B12'), ('B123','B1234','M4 added to B123'), ('B','B1234','Full versus B')]
    values = np.column_stack([pivot[p]-pivot[c] for p,c,_ in columns])
    fig, ax = plt.subplots(figsize=(9.6,8.4), constrained_layout=True)
    vmax = max(abs(values.min()),abs(values.max()))
    im = ax.imshow(values, cmap='RdBu', vmin=-vmax,vmax=vmax,aspect='auto')
    ax.set_xticks(range(len(columns)), [x[2] for x in columns],rotation=22,ha='right')
    ax.set_yticks(range(len(pivot)), [ds+': '+dom.replace('Tongji','T') for ds,dom in pivot.index])
    for (i,j),v in np.ndenumerate(values):
        ax.text(j,i,f'{v:+.2f}',ha='center',va='center',fontsize=8,color='white' if abs(v)>.6*vmax else '#111')
    ax.set_title('All 21 held-out conditions: stagewise MAE reduction (SOH pp)\nPositive values favor the added module; repeated development cohorts')
    fig.colorbar(im,ax=ax,shrink=.7,label='MAE reduction (pp)')
    for ext in ['png','pdf']: fig.savefig(OUT / ('domain_effects.'+ext), dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3,3,figsize=(12,9),constrained_layout=True)
    for ax, cell in zip(axes.flat, ['CS2_5','CS2_6','CX2_3','PL11','PL12','PL13','PL14']):
        for i, folder in enumerate(sorted((EXTERNAL / 'folds').iterdir())):
            for group, color in [('B','#8a8a8a'), ('B1234','#237b78')]:
                with np.load(folder/'predictions'/cell/(group+'.npz'),allow_pickle=False) as z:
                    ax.plot(z['cycle'],z['pred']*100,color=color,lw=.9,alpha=.5,
                            label=group+' (six source models)' if i==0 else None)
                    if i==0 and group=='B':
                        ax.plot(z['cycle'],z['y']*100,color='#161616',lw=1.0,alpha=.8,label='Observed query SOH')
        ax.set_title(cell); ax.set_xlabel('Recorded measurement index'); ax.set_ylabel('SOH (%)'); ax.grid(alpha=.12)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[7].axis('off'); axes.flat[7].legend(handles,labels,loc='center',frameon=False)
    axes.flat[8].axis('off')
    axes.flat[8].text(0,.65,'All seven eligible cells retained.\nEach source model is shown separately.\nNo prediction ensemble is evaluated.\nQuery traces follow the ten-label support.',va='top',fontsize=10,linespacing=1.6)
    fig.suptitle('Frozen CALCE predictions: nearly flat estimates despite observed capacity changes',fontsize=13)
    for ext in ['png','pdf']: fig.savefig(OUT / ('calce_trajectories.'+ext),dpi=180)
    plt.close(fig)
figures(pd.read_csv(E/'stage_cost.csv'),pd.read_csv(E/'domain_ablation.csv'),pd.read_csv(E/'calce_protocols.csv'))
print('Figures written to',OUT)

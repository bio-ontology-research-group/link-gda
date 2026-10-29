"""Rank CDFs from retained metrics; equally weight each supplied fold or seed."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot(spec,out):
    fig,axes=plt.subplots(1,2,figsize=(9,3.6),layout='constrained')
    head_max=0.;candidate_count=None
    for item in spec['series']:
        curves=[];n=None
        for file in item['files']:
            data=json.loads(Path(file).read_text());result=data['settings'][item['setting']]
            this_n=result['unfiltered']['candidates']
            if n is not None and n!=this_n:raise ValueError('Candidate counts differ within plotted series')
            n=this_n;ranks=np.sort(result['ranks'])
            grid=np.unique(np.r_[1,np.geomspace(1,n,500),10,100,n])
            curves.append(np.searchsorted(ranks,grid,side='right')/len(ranks))
        if candidate_count is not None and candidate_count!=n:raise ValueError("Plotted candidate pools have different sizes")
        candidate_count=n
        mean=np.mean(curves,axis=0)
        head_max=max(head_max,float(mean[grid<=100].max()))
        for ax in axes:ax.plot(grid,mean,label=item['label'],linestyle=item.get('linestyle','-'),linewidth=1.6)
    axes[0].set_xscale('log');axes[0].set_xlim(1,100);axes[0].set_title('Top 100');axes[0].set_ylim(0,min(1,max(.05,head_max*1.1)))
    axes[1].set_xscale('log');axes[1].set_title('Full candidate pool')
    for ax in axes:
        ax.set_xlabel('Gene rank');ax.set_ylabel('Fraction of associations retrieved');ax.grid(alpha=.2)
        ax.spines[['top','right']].set_visible(False)
    axes[1].legend(fontsize=8,loc='upper left')
    fig.suptitle(spec['title'],fontsize=11)
    fig.savefig(out);fig.savefig(Path(out).with_suffix('.png'),dpi=160)
    plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();plot(json.loads(a.spec.read_text()),a.out)

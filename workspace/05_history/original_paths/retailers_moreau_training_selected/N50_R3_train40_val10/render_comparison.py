"""Presentation-only redraw of saved plot_data; no evaluation or selection."""
from pathlib import Path
import time
import update_comparison as u
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

if __name__ == '__main__':
    start=time.perf_counter()
    started=u.c.utc()
    data=u.c.read_json(u.OUTPUT/'plot_data.json')
    fig,ax=plt.subplots(figsize=(11,7.5))
    colors=['#2166ac','#762a83','#111111','#009e73','#009e73','#d62728']
    markers=['o','o','D','^','^','s']
    for i,curve in enumerate(data['curves']):
        values=np.asarray(curve['per_seed'])
        np.testing.assert_array_equal(values.mean(axis=1),curve['mean'])
        np.testing.assert_array_equal(np.percentile(values,20,axis=1),curve['p20'])
        np.testing.assert_array_equal(np.percentile(values,80,axis=1),curve['p80'])
        ax.plot(curve['radius'],curve['mean'],marker=markers[i],ls='--' if i==5 else '-',
                color=colors[i],label=curve['label'],ms=5,lw=2)
        ax.fill_between(curve['radius'],curve['p20'],curve['p80'],color=colors[i],alpha=.12)
    for label,value,color,style in data['horizontal']:
        ax.axhline(value,color=color,ls=style,label=label)
    ax.set_xscale('symlog',linthresh=.001,linscale=1)
    ax.set_xlim(-.00015,1.65)  # Display-only: avoid automatic negative-radius padding.
    ax.set_xticks([0,.001,.003,.01,.03,.1,.2,.4,.7,1.5])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x:g}'))
    ax.set(xlabel=r'Training radius $\rho$ (linear near zero; logarithmic above .001)',
           ylabel='Mean out-of-sample cost',title='R=3, total N=50: updated retailer comparison\nSame 10,000 scenarios (seed 9999); 20/80 seed bands, not confidence intervals')
    ax.grid(axis='y',alpha=.25)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.17),ncol=2,fontsize=8)
    fig.text(.5,.025,'*Mean-stop checkpoints recovered from saved Moreau prefixes; not a fresh historical-SCS rerun.\n'
        'Only complete 10-seed cohorts shown; mean-rule .001/.003/.01/.03 unavailable. Zero is nominal only.\n'
        'Terminal includes capped seed 9 at .7. Previous lines unchanged; training/selection protocols differ.\n'
        'Previously inspected evaluation draw: descriptive only. Cached bounds use different samples.',ha='center',fontsize=8)
    fig.subplots_adjust(bottom=.38)
    fig.savefig(u.OUTPUT/'matched_oos_updated.png',dpi=180,bbox_inches='tight')
    plt.close(fig)
    u.c.write_json(u.OUTPUT/'render_timing.json',dict(status='completed',start_utc=started,end_utc=u.c.utc(),
        wall_time_s=time.perf_counter()-start,timing_clock='time.perf_counter',
        scope='presentation only; no evaluation or selection',
        source_plot_data_sha256=u.c.sha(u.OUTPUT/'plot_data.json'),script_sha256=u.c.sha(Path(__file__)),
        figure_sha256=u.c.sha(u.OUTPUT/'matched_oos_updated.png')))
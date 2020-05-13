#!/usr/bin/python3 -B
import pickle

from matplotlib.ticker import StrMethodFormatter
from scipy.optimize import curve_fit

pickle.HIGHEST_PROTOCOL = pickle.DEFAULT_PROTOCOL

import argparse
import collections
import contextlib
import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib

PDF_NONE = {'Creator': None, 'Producer': None, 'CreationDate': None}

@contextlib.contextmanager
def subplots(n, **k):
    k.setdefault('ncols', 4)
    k.setdefault('nrows', math.ceil(n / k['ncols']))
    k.setdefault('figsize', (4.75 * k['ncols'], 1.9 * k['nrows']))
    k.setdefault('squeeze', False)
    matplotlib.rcParams['pdf.use14corefonts'] = True
    matplotlib.rcParams['font.family'] = 'sans'
    matplotlib.rcParams['axes.linewidth'] = 0.5
    matplotlib.rcParams['lines.linewidth'] = 0.5
    matplotlib.rcParams['patch.linewidth'] = 0.25
    matplotlib.rcParams['figure.subplot.left'] = 0.05
    matplotlib.rcParams['figure.subplot.right'] = 0.95
    matplotlib.rcParams['figure.subplot.bottom'] = 0.35 / k['figsize'][1]
    matplotlib.rcParams['figure.subplot.top'] = 1.0 - 0.35 / k['figsize'][1]
    matplotlib.rcParams['figure.subplot.wspace'] = 0.35
    matplotlib.rcParams['figure.subplot.hspace'] = 0.3
    matplotlib.rcParams['axes.labelsize'] = 8
    matplotlib.rcParams['axes.titlesize'] = 8
    matplotlib.rcParams['axes.titleweight'] = 'medium'
    matplotlib.rcParams['xtick.labelsize'] = 8
    matplotlib.rcParams['ytick.labelsize'] = 8
    matplotlib.rcParams['xtick.major.width'] = 0.5
    matplotlib.rcParams['xtick.minor.width'] = 0.25
    matplotlib.rcParams['ytick.major.width'] = 0.5
    matplotlib.rcParams['ytick.minor.width'] = 0.25
    matplotlib.rcParams['axes.xmargin'] = 0.0
    matplotlib.rcParams['legend.fontsize'] = 8
    matplotlib.rcParams['legend.frameon'] = False
    if k.pop('latex', False):
        matplotlib.rcParams['text.usetex'] = True
        matplotlib.rcParams['font.family'] = 'sans-serif'
    fig, axes = plt.subplots(**k)
    try:
        yield fig, axes
    finally:
        plt.close(fig)
        matplotlib.rcdefaults()

def load_csv(path):
    dtype = {
        'seed' : np.uint8,
        'topo_num': np.uint8,
        'nodes': np.uint16,
        'edges': np.uint16,
        'units': np.uint16,
        'mean_demand': np.uint16,
        'n': np.uint32,
        'bad': np.uint32,
        'cum_demand': np.uint32,
        'cum_util': np.uint32,
        'src': str,
        'dst': str,
        'demand': np.uint16,
        'paths': np.uint16,
        'path_len': np.uint16,
        'cu_start': np.uint16,
        'elapsed': np.uint64
    }

    dfs = collections.defaultdict(list)

    for csv in sorted(path.glob('**/result.csv')):
        seed = int(csv.parts[-3])
        algorithm = csv.parts[-4]
        runtime = csv.parts[-5]
        d = pd.read_csv(csv, index_col=None, header=0, sep=' ', dtype=dtype)
        assert (d['seed'] == seed).all()
        dfs[runtime, algorithm].append(d)
        print('loaded', runtime, algorithm, seed)

    df = {}
    for (runtime, algorithm), dl in dfs.items():
        df[runtime, algorithm] = pd.concat(dl)
        del dl[:]
        df[runtime, algorithm].to_hdf(path / f'{runtime}_{algorithm}.hdf', key=f'{runtime}_{algorithm}', mode='w')
        print('concat', runtime, algorithm)

    return df

def load_data(path):
    df = {}
    for hdf in sorted(path.glob('*.hdf')):
        d = pd.read_hdf(hdf)
        d['elapsed'] = d['elapsed'].astype(float) / 1000000000
        d['usage'] = d['cum_util'] / (d['edges'].astype(np.uint64) * d['units'])
        d['usage'] = d['usage'].round(2)
        df[tuple(hdf.stem.split('_'))] = d

    return df

def plot_hist(ss, name, relative=False):
    fs = (13, 2.5)
    gs = {'wspace': 0.1, 'left': 0.045, 'right': 0.94, 'top': 0.98}
    bbox = dict(facecolor='w', edgecolor='none', boxstyle='square,pad=0.2')
    with subplots(0, ncols=len(ss), nrows=1, figsize=fs, gridspec_kw=gs, sharey='row', latex=True) as (fig, axes):
        axes = iter(axes.flatten())
        for s in ss:
            ax = next(axes)
            s = s.sort_values()
            s.index = np.linspace(0.0, 1.0, num=len(s), endpoint=False)

            s.plot(ax=ax, style='k')
            ax.set_yscale('log')
            x_start, x_end = [0.0, 1.0]
            y_start, y_end = [s.iloc[0] * 4, s.iloc[-1] / 12]

            if relative:
                prog_position = np.searchsorted(s, 1.0, side='left')
                y = s.iloc[prog_position]
                x = s.index[prog_position]
                ax.hlines(y=y, xmin=x_start, xmax=x, color='r', linestyle='--')
                ax.vlines(x=x, ymin=y_start, ymax=y, color='r', linestyle='--')
                print(x, y)
                ax.text(x_start + 0.03, y, f'{y:0.3f}', ha='left', va='center', fontsize=8, color='r', bbox=bbox)
                ax.text(x, y_start * 2, f'{x:0.3f}', ha='center', va='bottom', rotation='vertical', fontsize=8, color='r', bbox=bbox)
                ax.set_ylabel('Time [relative to filtered]')
            else:
                ax.set_ylabel('Time [seconds]')

            y = s.median()
            x = s.index[np.searchsorted(s, y, side='left')]
            ax.hlines(y=y, xmin=x_start, xmax=x, color='b', linestyle='--')
            ax.vlines(x=x, ymin=y_start, ymax=y, color='b', linestyle='--')
            ax.text(x_start + 0.03, y, f'{y:0.3f} (median)', ha='left', va='center', fontsize=8, color='b', bbox=bbox)
            ax.text(x, y_start * 2, f'{x:0.3f}', ha='center', va='bottom', rotation='vertical', fontsize=8, color='b', bbox=bbox)
            print(x, y)

            y = s.mean()
            x = s.index[np.searchsorted(s, y, side='left')]
            ax.hlines(y=y, xmin=x_start, xmax=x, color='g', linestyle='--')
            ax.vlines(x=x, ymin=y_start, ymax=y, color='g', linestyle='--')
            ax.text(x_start + 0.03, y, f'{y:0.3f} (mean)', ha='left', va='center', fontsize=8, color='g', bbox=bbox)
            ax.text(x, y_start * 2, f'{x:0.3f}', ha='center', va='bottom', rotation='vertical', fontsize=8, color='g', bbox=bbox)
            print(x, y)

            ax.set_xlim(x_start, x_end)
            ax.set_ylim(y_start, y_end)

            ax.set_xlabel('Fraction of calls')
            ax.yaxis.set_tick_params(labelleft=True)

        # fig.savefig(f'{name}_cdf.png', metadata=PDF_NONE)
        fig.savefig(f'{name}_cdf.pdf', metadata=PDF_NONE)

def plot_fit(ax, z, fc):
    if fc == 'n':
        def func(x, a, b, c):
            return a * x + c
    elif fc == 'n^2':
        def func(x, a, b, c):
            return a * (x ** 2) + c
    elif fc == 'n\:log\,n':
        def func(x, a, b, c):
            return a * x * np.log10(x) + c
    elif fc == 'log\,n':
        def func(x, a, b, c):
            return a * np.log10(x) + c
    else:
        raise NotImplementedError
    try:
        popt, pcov = curve_fit(func, z.index, z)
        residuals = z.values - func(z.index, *popt)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((z.values - np.mean(z.values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        ax.plot(z.index, func(z.index, *popt), label=f"${fc}\ (R^2={r_squared:0.6f})$")
    except RuntimeError:
        pass

def plot_groups(s, name, relative=False):
    fs = (13, 2.5)
    gs = {'wspace': 0.14, 'left': 0.045, 'right': 0.94, 'top': 0.98}
    pathlib.Path(name).parent.mkdir(parents=True, exist_ok=True)
    with subplots(0, ncols=3, nrows=1, figsize=fs, gridspec_kw=gs, latex=True) as (fig, axes):
        axes = iter(axes.flatten())
        ax = next(axes)
        z = s.groupby('nodes').mean()['elapsed']
        z.plot(ax=ax, style='k.-', ms=4)
        ax.margins(x=0.05)
        ax.autoscale(tight=False)
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.3f}'))
        ax.set_xticks(z.index)
        ax.set_ylabel('Time [relative to filtered]' if relative else 'Time [seconds]')
        ax.set_xlabel('Number of nodes')
        if not relative:
            if 'filtered' in name:
                plot_fit(ax, z, 'n')
                plot_fit(ax, z, 'n\:log\,n')
            if 'generic' in name:
                plot_fit(ax, z, 'n^2')
        ax.legend()

        ax = next(axes)
        z = s.groupby('units').mean()['elapsed']
        z.plot(ax=ax, style='k.-', ms=4)
        ax.margins(x=0.05)
        ax.autoscale(tight=False)
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.3f}'))
        ax.set_xticks(z.index)
        # ax.set_ylabel('Time [relative to filtered]' if relative else 'Time [seconds]')
        ax.set_xlabel('Number of edge units')
        if not relative:
            if 'filtered' in name:
                plot_fit(ax, z, 'n')
            if 'generic' in name:
                plot_fit(ax, z, 'n')
                plot_fit(ax, z, 'log\,n')
        ax.legend()

        ax = next(axes)
        z = s[s['usage'] <= 0.6].groupby('usage').mean()['elapsed']
        z.plot(ax=ax, style='k.-', ms=4)
        ax.margins(x=0.05)
        ax.autoscale(tight=False)
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.3f}'))
        # ax.set_ylabel('Time [relative to filtered]' if relative else 'Time [seconds]')
        ax.set_xlabel('Network utilization')
        ax.legend()

        # fig.savefig(f'{name}_groups.png', metadata=PDF_NONE)
        fig.savefig(f'{name}_groups.pdf', metadata=PDF_NONE)

def check_same_results(df):
    last_d = None
    cols = None
    for d in df.values():
        if last_d is not None:
            assert d[cols].equals(last_d[cols])
        else:
            cols = d.columns.difference(['elapsed'])
        last_d = d

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results_dir', nargs='?', default='')
    app_args = parser.parse_args()

    path = pathlib.Path(app_args.results_dir)

    df = load_data(path)
    print([len(d) for d in df.values()])

    pd.set_option('float_format', '{:.6f}'.format)

    for result in ['all']:
        if result == 'all':
            dd = df
        elif result == 'good':
            dd = {k: v[v['path_len'] > 0] for k, v in df.items()}
        elif result == 'bad':
            dd = {k: v[v['path_len'] == 0] for k, v in df.items()}
        else:
            raise ValueError

        for k, v in dd.items():
            plot_groups(v, result + '/' + '_'.join(k))
            plot_hist([v['elapsed']], result + '/' + '_'.join(k))

        ss = []
        for interpreter in ['python3', 'pypy3']:
            d = dd[interpreter, 'generic'].copy()
            d['elapsed'] /= dd[interpreter, 'filtered']['elapsed']
            plot_groups(d, result + '/' + f'{interpreter}_relative_to_filtered', relative=True)
            ss.append(d['elapsed'])

        plot_hist(ss, result + '/' + f'relative_to_filtered', relative=True)


if __name__ == '__main__':
    main()

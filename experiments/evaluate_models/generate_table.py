"""
Generate the main results LaTeX table.
Run from project root: python experiments/evaluate_models/generate_table.py
"""
import glob, numpy as np, joblib, sys

def lp(l):
    return '1' if l == 1.0 else str(l)

def load_vals(folder, mdir, keys):
    path = folder + '/' + mdir
    splits = sorted(glob.glob(path + '/*_split_*_results.joblib'))
    out = {k: [] for k in keys}
    for s in splits:
        d = joblib.load(s)
        for k in keys:
            out[k].append(d.get(k, np.nan))
    return {k: np.array(v) for k, v in out.items()}

KEYS = ['test_acc_y', 'test_acc_c', 'test_ctl_average', 'test_icl_average']
R = 'results/'
LAM = [0.1, 0.5, 1.0]

MODELS = {
    'Joint CBM': {
        'tt':  {l: (R+'tabulartoy_25_10k_models', 'SoftCBM_adam_lr0.05_bs64_lam_c'+lp(l)) for l in LAM},
        'ds':  {l: (R+'dsprites_dep_0_models',    'SoftCBM_adam_lr1e-04_bs64_lam_c'+lp(l)) for l in LAM},
        'cub': {l: (R+'cub_soft_0.01',            'SoftCBM_adam_lr1e-04_bs256_lam_c'+lp(l)) for l in LAM},
    },
    'Sequential CBM': {
        'tt':  {1.0: (R+'tabulartoy_25_10k_models_acbm_shared_critic', 'SeqCBM_adam_lr0.05_bs64_lam_c1')},
        'ds':  {1.0: (R+'dsprites_sequential_acbm', 'SeqCBM_adam_lr1e-04_bs64_lam_c1')},
        'cub': {1.0: (R+'cub_acbm_shared_critic',   'SeqCBM_adam_lr1e-04_bs256_lam_c1')},
    },
    'Hard CBM': {
        'tt':  {1.0: (R+'tabulartoy_25_10k_models', 'HardCBM_adam_lr0.05_bs64_lam_c1')},
        'ds':  {1.0: (R+'dsprites_dep_0_models',    'HardCBM_adam_lr1e-04_bs64_lam_c1')},
        'cub': {1.0: (R+'cub_soft_0.01',            'HardCBM_adam_lr1e-04_bs256_lam_c1')},
    },
    'GC-CBM': {
        'tt':  {l: (R+'tabulartoy_25_10k_models_acbm_shared_critic', 'CRCBM_adam_lr0.01_bs64_lam1_none_lam_c'+lp(l)+'_shared_critic') for l in LAM},
        'ds':  {l: (R+'dsprites_ACBM_shared_critic', 'CRCBM_adam_lr5e-05_bs64_lam1_none_lam_c'+lp(l)+'_shared_critic') for l in LAM},
        'cub': {l: (R+'cub_acbm_shared_critic',      'ACBM_adam_lr1e-04_bs256_lam1_none_lam_c'+lp(l)+'_shared_critic') for l in LAM},
    },
    'Joint CEM': {
        'tt':  {},
        'ds':  {l: (R+'dsprites_cem', 'CEM_adam_lr1e-03_bs64_lam_c'+lp(l)) for l in LAM},
        'cub': {l: (R+'cub_cem',      'CEM_adam_lr1e-03_bs256_lam_c'+lp(l)) for l in LAM},
    },
    'GC-CEM': {
        'tt':  {l: (R+'tabulartoy_25_10k_models_acem_shared_critic', 'ACEM_adam_lr1e-03_bs64_lam1_none_lam_c'+lp(l)+'_shared_critic') for l in LAM},
        'ds':  {l: (R+'dsprites_acem_shared_critic', 'CRCEM_adam_lr5e-05_bs64_lam1_none_lam_c'+lp(l)+'_shared_critic') for l in LAM},
        'cub': {l: (R+'cub_cem',                     'CRCEM_adam_lr5e-04_bs256_lam1_none_lam_c'+lp(l)+'_shared_critic') for l in LAM},
    },
}

# Pre-compute means for bolding
avgs = {}
for mname, mdata in MODELS.items():
    for ds in ['tt', 'ds', 'cub']:
        for lam in LAM:
            if lam not in mdata[ds]:
                continue
            folder, mdir = mdata[ds][lam]
            d = load_vals(folder, mdir, KEYS)
            for key in KEYS:
                v = d[key][~np.isnan(d[key])]
                if len(v):
                    avgs[(mname, ds, lam, key)] = np.mean(v)


def best_model(ds, lam, key):
    cands = {m: v for (m, d, l, k), v in avgs.items()
             if d == ds and l == lam and k == key}
    if not cands:
        return None
    return (min(cands, key=lambda m: cands[m]) if 'ctl' in key
            else max(cands, key=lambda m: cands[m]))


def cell(mname, ds, lam, key, pct=True):
    if lam not in MODELS[mname][ds]:
        return '--'
    folder, mdir = MODELS[mname][ds][lam]
    d = load_vals(folder, mdir, KEYS)
    v = d[key][~np.isnan(d[key])]
    if not len(v):
        return '--'
    m, s = np.mean(v), np.std(v)
    ms = '{:.1f}'.format(m * 100) if pct else '{:.3f}'.format(m)
    ss = '{:.1f}'.format(s * 100) if pct else '{:.3f}'.format(s)
    if best_model(ds, lam, key) == mname:
        ms = r'\textbf{' + ms + '}'
    return ms + r' \pm ' + ss


lines = []
lines.append(r'\begin{table*}[h]')
lines.append(r'\caption{Task accuracy, concept accuracy, CTL and ICL across models, '
             r'datasets and $\lambda_c$ values. Results are mean $\pm$ std over 5 seeds. '
             r'Best results per dataset and $\lambda_c$ are \textbf{bolded}. '
             r'Task and concept accuracy are in \%.}')
lines.append(r'\label{tab:main_results}')
lines.append(r'\centering')
lines.append(r'\resizebox{\textwidth}{!}{%')
lines.append(r'\begin{tabular}{llcccccccccccc}')
lines.append(r'\toprule')
lines.append(r'& & \multicolumn{4}{c}{\textbf{TabularToy}} '
             r'& \multicolumn{4}{c}{\textbf{dSprites}} '
             r'& \multicolumn{4}{c}{\textbf{CUB}} \\')
lines.append(r'\cmidrule(lr){3-6} \cmidrule(lr){7-10} \cmidrule(lr){11-14}')
lines.append(r'\textbf{Model} & $\lambda_c$ '
             r'& Task $\uparrow$ & Con.\ $\uparrow$ & CTL $\downarrow$ & ICL $\downarrow$ '
             r'& Task $\uparrow$ & Con.\ $\uparrow$ & CTL $\downarrow$ & ICL $\downarrow$ '
             r'& Task $\uparrow$ & Con.\ $\uparrow$ & CTL $\downarrow$ & ICL $\downarrow$ \\')
lines.append(r'\midrule')

MODEL_LIST = list(MODELS.keys())
for mi, mname in enumerate(MODEL_LIST):
    ours = ' (ours)' if 'GC' in mname else ''
    label = mname + ours
    for ri, lam in enumerate(LAM):
        prefix = r'\multirow{3}{*}{' + label + '}' if ri == 0 else ''
        parts = [prefix, '$' + lp(lam) + '$']
        for ds in ['tt', 'ds', 'cub']:
            parts.append('$' + cell(mname, ds, lam, 'test_acc_y') + '$')
            parts.append('$' + cell(mname, ds, lam, 'test_acc_c') + '$')
            parts.append('$' + cell(mname, ds, lam, 'test_ctl_average', False) + '$')
            parts.append('$' + cell(mname, ds, lam, 'test_icl_average', False) + '$')
        lines.append(' & '.join(parts) + r' \\')
    if mi < len(MODEL_LIST) - 1:
        lines.append(r'\midrule')

lines.append(r'\bottomrule')
lines.append(r'\end{tabular}')
lines.append(r'}')  # close resizebox
lines.append(r'\end{table*}')

for line in lines:
    print(line)

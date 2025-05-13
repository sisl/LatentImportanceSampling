# packages
import json
import numpy as np
from tabulate import tabulate

OUTPUT_MODE = "NORMAL"

def calculate_mean_std(metric_values, correction, fmt):
    trimmed_values = np.array(np.trim_zeros(metric_values,"b"))
    if fmt == "f":
        if OUTPUT_MODE == "LATEX":
            return rf"{np.nanmean(correction*trimmed_values):.3f} \pm {np.nanstd(correction*trimmed_values):.3f}"
        else:
            return rf"{np.nanmean(correction*trimmed_values):.3f} +- {np.nanstd(correction*trimmed_values):.3f} ({trimmed_values.shape[0]}, {np.isnan(trimmed_values).sum().item()})" 
    elif fmt == "i":
        if OUTPUT_MODE == "LATEX":
            return rf"{int(np.nanmean(correction*trimmed_values)):d} \pm {int(np.nanstd(correction*trimmed_values)):d}"
        else:
            return rf"{int(np.nanmean(correction*trimmed_values)):d} +- {int(np.nanstd(correction*trimmed_values)):d} ({trimmed_values.shape[0]}, {np.isnan(trimmed_values).sum().item()})" 

metrics = ["rel errs", "log probs", "coverages", "densities", "total samples"]
table_headers = ["rel errs", "NLL", "coverages", "densities", "total samples"]
metrics_correction = [1, -1, 1, 1, 1]
metrics_format = ["f", "f", "f", "f", "i"]
methods = ["latent-ce", "latent-sis", "target-ce", "target-sis"]
systems = ["robot", "racecar", "f16"]
n_trial_per_system = [100, 100, 20]

results_path = "./results/"
experiments = {s:{m:{} for m in methods} for s in systems}
for s in systems:
    for m in methods:
        path = results_path+s+"/"+m+"/results.json"
        try:
            experiments[s][m] = json.load(open(path, "r"))
        except:
            experiments[s][m] = None



for n_trials, e in zip(n_trial_per_system,experiments):
    print(f"Results for experiment: {e}")
    
    header_row = ["", ] + table_headers
    result_rows = []
    
    if OUTPUT_MODE == "LATEX":
        for m in methods:
            if experiments[e][m] is not None:
                result_rows.append([""]+[m]+[calculate_mean_std(experiments[e][m][met][:n_trials], metrics_correction[i], metrics_format[i]) for i,met in enumerate(metrics)])
                
            else:
                result_rows.append([""]+[m]+["---" for met in metrics])
        
        best_val_fun = [lambda x: np.argmin(np.abs(x)), np.argmin, np.argmax, np.argmax, np.argmin]
        vals = np.zeros((4,5))
        for j in range(2,7):
            for i in range(4):
                vals[i,j-2] = float(result_rows[i][j].split(r"\pm")[0])
        vals = np.array(vals)
        
        best_idx = []
        for i, bvf in enumerate(best_val_fun):
            bidx = bvf(vals[:,i])
            result_rows[bidx][i+2] = r"\bfseries "+result_rows[bidx][i+2]
        print(tabulate(result_rows, headers=header_row, tablefmt="latex_raw"))
            
    else:
        for m in methods:
            if experiments[e][m] is not None:
                result_rows.append([m]+[calculate_mean_std(experiments[e][m][met][:n_trials], metrics_correction[i], metrics_format[i]) for i,met in enumerate(metrics)])
            else:
                result_rows.append([m]+["None" for met in metrics])
        print(tabulate(result_rows, headers=header_row, tablefmt="grid"))
        
    
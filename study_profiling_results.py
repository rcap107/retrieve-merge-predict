#%%
import json
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
# %%
respath = Path("results/profiling/retrieval/exact_matching/binary_update")
# %%
respath
# %%
temp = json.load(open("results/profiling/retrieval/exact_matching/binary_update/2025-03-05T15-11-40.json"))
# %%
def parse_results(log_dict):
    mem_create = np.array(log_dict["memory_usage"]["create"])
    mem_query = np.array(log_dict["memory_usage"]["query"])
    
    mem_trace = np.vstack([mem_create, mem_query])
    mem_trace[:, [0, 1]] = mem_trace[:, [1, 0]]

    return mem_trace
# %%
mem_trace = parse_results(temp)
# %%
plt.plot(mem_trace[:,0], mem_trace[:,1])
# %%
traces = []

for pth in respath.iterdir():
    d  = json.load(open(pth))
    trace = parse_results(d)
    traces.append(trace)
    
# %%
maxima = []
for t in traces:
    plt.plot(range(len(t)), t[:,1])
    maxima.append(t[:,1].max())
# %%

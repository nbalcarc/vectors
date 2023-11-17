#import numpy as np
import pandas as pd
import os
#from functools import reduce

def run_stats() -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    all_files = os.listdir("inputs/datasets")
    cultivar_files = list(map(lambda x: "inputs/datasets/" + x, all_files))
    cultivars = list(map(lambda x: x[x.rfind("_")+1:x.rfind(".")], all_files))

    # first collect all phenology tables and find all unique phenologies
    phenologies: dict[str, pd.Series] = dict()
    all_phenologies: set[str] = set()

    for i in range(len(cultivar_files)):
        df: pd.DataFrame = pd.read_csv(cultivar_files[i])
        phenologies[cultivars[i]] = df["PHENOLOGY"] #panda series
        all_phenologies |= set(df["PHENOLOGY"].dropna()) #union of two sets

    # collect stats for each cultivar
    cultivar_stats: dict[str, dict[str, int]] = dict() #cultivar; phenology; count

    for k in cultivars:
        cur_phens: list[str] = list(phenologies[k].dropna()) #phenologies for current cultivar
        phen_stats: dict[str, int] = dict()
        for p in all_phenologies:
            phen_stats[p] = cur_phens.count(p)
        cultivar_stats[k] = phen_stats

    # now generate overall stats
    all_stats = {k:0 for k in all_phenologies} #phenology; count
    z = list(map(lambda x: list(x.items()), cultivar_stats.values())) #get every cultivar's phenology data
    for p in z[1:]: #concatenate all phenology data into one list
        z[0].extend(p)
    for t in z[0]: #sum all counts
        all_stats[t[0]] += t[1]

    return cultivar_stats, all_stats


def main():
    """Main entry point."""
    cultivar_stats, all_stats = run_stats()
    all_stats_list = list(all_stats.items())
    all_stats_list.sort(key = lambda k: k[1], reverse = True)

    cultivar_names = list(cultivar_stats.keys())
    shared_phens = set(cultivar_stats[cultivar_names[0]].keys()) #list of phenology candidates
    for cultivar in cultivar_stats.keys():
        dicti_items = cultivar_stats[cultivar].items()
        dicti = list(filter(lambda x: x[1] != 0, dicti_items)) #ignore phens with 0 occurences
        if len(dicti) == 0: #if no phens recorded then skip this cultivar
            continue
        phens = list(map(lambda x: x[0], dicti)) #map to only the phenology names (no counts)
        shared_phens.intersection_update(set(phens)) #intersect
    trimmed_stats = dict()
    for i in shared_phens:
        trimmed_stats[i] = all_stats[i]
    trimmed_stats_list = list(trimmed_stats.items())
    trimmed_stats_list.sort(key = lambda k: k[1], reverse = True)
    print(trimmed_stats_list)



    #print(all_stats)

if __name__ == "__main__":
    main()








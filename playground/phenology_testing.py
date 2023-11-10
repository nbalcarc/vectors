import numpy as np
import pandas as pd
import os

def run_stats():
    all_files = os.listdir("inputs/datasets")
    cultivar_files = list(map(lambda x: "inputs/datasets/" + x, all_files))
    cultivars = list(map(lambda x: x[x.rfind("_")+1:x.rfind(".")], all_files))
    print(cultivar_files)
    print(cultivars)

    phenologies = dict()

    for i in range(len(cultivar_files)):
        df = pd.read_csv(cultivar_files[i])
        phenologies[cultivars[i]] = df["PHENOLOGY"]

    print(phenologies["Riesling"])


def main():
    """Main entry point."""
    run_stats()
    pass

if __name__ == "__main__":
    main()








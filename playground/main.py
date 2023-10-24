import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import data
import predict
import testing
import compute
import columns as col


'''
Notes:

state_vectors has only values between -1 and 1
with 2048 dimensions, this means a vector can be a maximum of 45.254833995939045 in length (sqrt of 2048)

want to output graphs for seasons 2002-2003 through 2011-2012, which are indices 10 through 19

want to compute different clusters of points, and these clusters will color the points on the graph
    - a cluster could be revisted later theoretically, so it will have two+ areas of the graph of the same color

consider splitting data module into subtree of io, functions, etc

'''


def run_task_embedded():
    data.load_data_embedded()
    pass


def run_multihead():
    season_names = [
        "2002-2003",
        "2003-2004",
        "2004-2005",
        "2005-2006",
        "2006-2007",
        "2007-2008",
        "2008-2009",
        "2009-2010",
        "2010-2011",
        "2011-2012",
        ]

    # retrieve phenology data
    phenology_df = data.get_phenology_dataframe()

    # run calculations
    seasons = range(10, 20)
    dbscan = compute.dbscan(seasons, eps = 3.0, min_samples = 4)
    cluster_max, cluster_avg = compute.cluster_stats(seasons, dbscan)
    print(len(cluster_max))
    print(len(cluster_avg))
    
    k5 = compute.k_span(seasons, 5)
    k10 = compute.k_span(seasons, 10)

    '''
    at the end maybe plot all the dbscan results overlaid, each with a low opacity

    plot the cluster data in an table maybe (try using pandas and converting to a matplotlib table)


    '''

    rows = ["Max", "Avg"]

    # graph
    for s in seasons: #for each season
        
        cur_season = season_names[s-10]
        cur_phenologies = phenology_for_season(phenology_df, cur_season)

        # organize cluster stats
        num_clusters = cluster_max[s-10].shape[0]
        cur_stats = np.zeros((3, num_clusters))
        cur_stats[0,:] = cluster_max[s-10] #load the maxs
        cur_stats[1,:] = cluster_avg[s-10] #load the avgs
        cur_stats[2,:] = cur_stats[0,:] / cur_stats[1,:] #calculate the factors
        df_stats = pd.DataFrame(cur_stats) #easy to generate a table from a pandas dataframe so we do that conversion here

        # output table cluster stats
        fig, ax = plt.subplots()
        fig.patch.set_visible(False)
        ax.axis("off")
        ax.axis("tight")
        table = ax.table(cellText=df_stats.values, colLabels=df_stats.columns, rowLabels=["Max", "Avg", "Factor"], loc="center")
        fig.tight_layout()
        plt.savefig("output_graphs/cluster_stats_" + cur_season + ".png", dpi=300)

        # output graph dbscan
        plt.close()
        plt.clf()
        plt.figure(figsize = (6.4, 4.8), dpi = 100)
        plt.plot(list(range(1,251)), dbscan[s-10])
        plt.title(f"DBSCAN {cur_season}")
        insert_phenology(cur_phenologies, 0.5)
        plt.savefig("output_graphs/dbscan_" + cur_season + ".png")

        # output table clusters
        cur_max = cluster_max[s-10]
        cur_avg = cluster_avg[s-10]
        columns_num = len(cur_max)
        columns = [f"Cluster {i}" for i in range(columns_num - 1)]
        plt.close()
        plt.clf()
        plt.table(cellText = "ab", rowLabels = rows, colLabels = columns)
        plt.savefig(f"output_graphs/test_table.png")

        # output graph k-span 5
        plt.close()
        plt.clf()
        plt.figure(figsize = (6.4, 4.8), dpi = 100)
        plt.plot(list(range(1,246)), k5[s-10])
        plt.title(f"K-span(5) {cur_season}")
        insert_phenology(cur_phenologies, 3)
        plt.savefig(f"output_graphs/kspan5_" + cur_season + ".png")

        # output graph k-span 10
        plt.close()
        plt.clf()
        plt.figure(figsize = (6.4, 4.8), dpi = 100)
        plt.plot(list(range(1,241)), k10[s-10])
        plt.title(f"K-span(10) {cur_season}")
        insert_phenology(cur_phenologies, 3)
        plt.savefig(f"output_graphs/kspan10_" + cur_season + ".png")

        # output graph k-span 5 overlaid with dbscan
        plt.close()
        plt.clf()
        plt.figure(figsize = (6.4, 4.8), dpi = 100)
        plt.plot(list(range(1,246)), k5[s-10], label = "K5")
        plt.plot(list(range(1,251)), dbscan[s-10], label = "DB")
        plt.legend(loc="upper left")
        plt.title(f"K-span(5) with DBSCAN {cur_season}")
        insert_phenology(cur_phenologies, 3)
        plt.savefig(f"output_graphs/dbscan_kspan5_" + cur_season + ".png")

        # output graph k-span 10 overlaid with dbscan
        plt.close()
        plt.clf()
        plt.figure(figsize = (6.4, 4.8), dpi = 100)
        plt.plot(list(range(1,241)), k10[s-10], label = "K10")
        plt.plot(list(range(1,251)), dbscan[s-10], label = "DB")
        plt.legend(loc="upper left")
        plt.title(f"K-span(10) with DBSCAN {cur_season}")
        insert_phenology(cur_phenologies, 3)
        plt.savefig(f"output_graphs/dbscan_kspan10_" + cur_season + ".png")

    # output graph all clusters
    plt.close()
    plt.clf()
    plt.figure(figsize = (6.4, 4.8), dpi = 100)
    for s in seasons: #for each season
        cur_season = season_names[s-10]
        cur_phenologies = phenology_for_season(phenology_df, cur_season)
        plt.plot(list(range(1,251)), dbscan[s-10], alpha = 0.2)
    plt.title(f"DBSCAN All Seasons")
    plt.savefig("output_graphs/dbscan_all.png")


def phenology_for_season(df: pd.DataFrame, season: str) -> pd.DataFrame:
    """Grab the phenology data for the current season"""
    cur: pd.DataFrame = df[df[col.SEASON] == season] #filter to only the current season
    dorm = cur[cur[col.DORMANT_SEASON] == 1].copy() #filter to only within the dormancy season
    dorm[col.PHENOLOGY].fillna(0, inplace=True)
    clean = dorm[dorm[col.PHENOLOGY] != 0] #filter out NaNs
    short: pd.DataFrame = clean[[col.PHENOLOGY, col.DORMANT_DAY]] #only care about phenology and the day

    return short


def insert_phenology(phenologies: pd.DataFrame, y_coordinate: float):
    """Adds phenology data to the grpah, asks for a y-coordinate of the labels"""
    for row in phenologies.iterrows():
        plt.axvline(row[1][1], color = "red") #graph at the specified index
        plt.text(row[1][1], y_coordinate, row[1][0], rotation=90, alpha=0.5) #add a phenology label


def main():
    """Main entry point."""
    run_task_embedded()
    pass

if __name__ == "__main__":
    main()



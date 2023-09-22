import matplotlib.pyplot as plt
import pandas as pd

import data
import predict
import testing
import compute
import columns as col


seasons = [
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


def main():
    """Main entry point."""
    #testing.torch_testing()
    #testing.torch_testing_regression()

    #data.save_data_interoperable()
    #data.save_data()

    phenology_df = data.get_phenology_dataframe()

    l2, cos = compute.similarity()
    dbscan = compute.dbscan()
    k5 = compute.k_span(5)
    k10 = compute.k_span(10)

    print(l2.shape)
    print(cos.shape)
    print(dbscan.shape)
    print(k5.shape)
    print(k10.shape)

    for s in range(10, 20): #for each season
        cur_season = seasons[s-10]
        cur_phenologies = phenology_for_season(phenology_df, cur_season)

        # output graph euclidean / l2
        plt.close()
        plt.clf()
        plt.figure(figsize = (6.4, 4.8), dpi = 100)
        plt.plot(list(range(1, 250)), l2[s-10])
        plt.title("Euclidean Distances " + cur_season)
        insert_phenology(cur_phenologies, 0.5)
        plt.savefig("output_graphs/euclidean_" + cur_season + ".png")

        # output graph cosine similarity
        plt.close()
        plt.clf()
        plt.figure(figsize = (6.4, 4.8), dpi = 100)
        plt.plot(list(range(1, 250)), cos[s-10])
        plt.title("Cosine Similarities " + cur_season)
        insert_phenology(cur_phenologies, 1450)
        plt.savefig("output_graphs/cosine_" + cur_season + ".png")

        # output graph dbscan
        plt.close()
        plt.clf()
        plt.figure(figsize = (6.4, 4.8), dpi = 100)
        plt.plot(list(range(1,251)), dbscan[s-10])
        #plt.title(f"DBSCAN {cur_season} eps={eps}, min_samples={min_samples}")
        plt.title(f"DBSCAN {cur_season}")
        insert_phenology(cur_phenologies, 0.5)
        plt.savefig("output_graphs/dbscan_" + cur_season + ".png")

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





def phenology_for_season(df: pd.DataFrame, season: str) -> pd.DataFrame:
    """Grab the phenology data for the current season"""
    cur: pd.DataFrame = df[df[col.SEASON] == season] #filter to only the current season
    dorm = cur[cur[col.DORMANT_SEASON] == 1].copy() #filter to only within the dormancy season
    dorm[col.PHENOLOGY].fillna(0, inplace=True)
    clean = dorm[dorm[col.PHENOLOGY] != 0] #filter out NaNs
    short: pd.DataFrame = clean[[col.PHENOLOGY, col.DORMANT_DAY]] #only care about phenology and the day

    return short



# adds phenology data to the graph, asks for a y-coordinate of the labels
def insert_phenology(phenologies: pd.DataFrame, y_coordinate: float):
    for row in phenologies.iterrows():
        plt.axvline(row[1][1], color = "red") #graph at the specified index
        plt.text(row[1][1], y_coordinate, row[1][0], rotation=90, alpha=0.5) #add a phenology label



if __name__ == "__main__":
    main()



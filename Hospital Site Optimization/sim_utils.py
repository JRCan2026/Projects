import osmnx as ox
import time
import math
import networkx as nx
import numpy as np
import pandas as pd
from pyomo.environ import *
from pyomo.environ import exp
import matplotlib.pyplot as plt
import folium
from matplotlib import colors
from pyomo.environ import value
from folium.plugins import HeatMap
from concurrent.futures import ProcessPoolExecutor


# ------------------------------------------------------------------
# SCALER [0, 1] 
# ------------------------------------------------------------------
def normalize(column):
    return (column - column.min()) / (column.max() - column.min())

# ------------------------------------------------------------------
# MEAN TIME METRIC  
# ------------------------------------------------------------------
def compute_mean_time(T, spots, model, candidates, return_series=False):

    # Ensure DataFrame format
    T = pd.DataFrame(T, index=spots.index, columns=candidates.index)

    # Select open hospitals
    idx = [j for j in model.J if value(model.x[j]) > 0]

    if len(idx) == 0:
        raise ValueError("No hospitals selected")

    # Filter matrix
    T_open = T.loc[:, idx]

    # Compute minimum travel time
    spots = spots.copy()
    spots["min_time"] = T_open.min(axis=1)

    # Mean travel time 
    mean_time = spots["min_time"].mean()

    return round(mean_time,1)


# ------------------------------------------------------------------
# EFFICIENCY METRIC  
# ------------------------------------------------------------------
def compute_efficiency(model):
    # baseline risk: vulnerability when there'are no hospitals
    R0 = sum(model.n[i] for i in model.I)
    
    # Posteriori risk: remining vulnerability after optimization 
    R = sum(model.n[i] * np.exp(-value(model.C[i])) for i in model.I)

    # Efficiency
    eta = 1 - (R / R0)
    eta = eta * 100

    return round(eta,2)


# ------------------------------------------------------------------
# UNPROTECTED RATIO  
# ------------------------------------------------------------------
def compute_unprotected_ratio(model, spots, threshold=1e-6):

    coverage = np.array([value(model.C[i]) for i in model.I])

    unprotected = coverage <= threshold

    return unprotected.sum() * 100 / len(coverage)

# ------------------------------------------------------------------
# GINI INDEX  
# ------------------------------------------------------------------
def compute_gini(x):
    x = np.array(x, dtype=float)

    if np.amin(x) < 0:
        x -= np.amin(x)  # shift if negatives

    x = np.sort(x)
    n = len(x)

    if np.sum(x) == 0:
        return 0.0

    index = np.arange(1, n + 1)

    gini = (2 * np.sum(index * x)) / (n * np.sum(x)) - (n + 1) / n

    return gini

# ------------------------------------------------------------------
# ROUTINE COMPUTING THE TRAVEL TIME MATRIX 
# ------------------------------------------------------------------
def compute_travel_time_matrix(demand_df, facility_df, place_name="Quebec, Quebec, Canada", network_type="drive", weight="travel_time"):
    """
    Downloads the street network, projects demand and facility coordinates to the nearest nodes,
    and computes the OD (Origin-Destination) travel time matrix using Dijkstra's algorithm.
    """
    print(f"Downloading street network for: {place_name}...")
    G = ox.graph_from_place(place_name, network_type=network_type)
    
    # Add speeds and travel times  
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)

    print("Mapping coordinates to network nodes...")
    # Find nearest nodes to candidate hospotals
    demand_nodes = ox.distance.nearest_nodes(G, demand_df.geometry.x.values, demand_df.geometry.y.values)
    facility_nodes = ox.distance.nearest_nodes(G, facility_df['longitude'].values, facility_df['latitude'].values)

    print("Computing travel time matrix...")
    time_matrix = np.zeros((len(demand_nodes), len(facility_nodes)))

    for j, f in enumerate(facility_nodes):
        # optimal rute from hospital j to all existing nodes
        lengths = nx.single_source_dijkstra_path_length(G, f, weight=weight)


        for i, d in enumerate(demand_nodes):
            # if the node is unreacheble assign inf
            time_matrix[i, j] = lengths.get(d, np.inf) / 60

    print("Matrix computation completed successfully.")
    return time_matrix, G

# ------------------------------------------------------------------
# ROUTINE OPTIMIZING THE COVERAGE FUNCTION 
# ------------------------------------------------------------------
def coverage_model(spots, candidates, T, th):
    # Model 1: Linear Coverage
    model = ConcreteModel()
    
    model.I = Set(initialize=spots.index.tolist())
    model.J = Set(initialize=candidates.index.tolist())
    
    # Number of cases
    n_i = spots['events'].to_dict()
    
    # accessibility matrix
    gamma = 0.046 #math.log(0.5) / half_life
    w_ij = {
        (i, j): np.exp(-gamma * T[i][j]) if T[i][j] <= th else 0.0
        for i in model.I
        for j in model.J
    }
    
    model.w = Param(model.I, model.J, initialize=w_ij) # accecibility
    model.n = Param(model.I, initialize=n_i)
    model.x = Var(model.J, domain=Binary)
    
    # Expresion to define cumulative linear coverage
    def C(model, i):
        return sum(model.w[i,j]*model.x[j] for j in model.J)
    model.C = Expression(model.I, rule=C)
    
    # Objective function (total coverage)
    def coverage(model):
        return sum(model.n[i] * model.C[i] for i in model.I)
    model.obj = Objective(rule=coverage, sense=maximize)
    
    # budget-constraint (K hospitals)
    model.K = Param(initialize=1, mutable=True)
    
    def cap_rule(model):
        return sum(model.x[j] for j in model.J) <= model.K
    
    model.cap = Constraint(rule=cap_rule)
    
    # solver (create ONCE)
    solver = SolverFactory('glpk')

    return model, solver

# ------------------------------------------------------------------
# ROUTINE OPTIMIZING THE RISK FUNCTION 
# ------------------------------------------------------------------
def risk_model(spots, candidates, T, th):
    # Model 1: Linear Coverage
    model = ConcreteModel()
    
    model.I = Set(initialize=spots.index.tolist())
    model.J = Set(initialize=candidates.index.tolist())
    
    # Number of cases
    n_i = spots['events'].to_dict()
    
    # accessibility matrix
    gamma = 0.046 #math.log(0.5) / half_life
    w_ij = {
        (i, j): exp(-gamma * T[i][j]) if T[i][j] <= th else 0.0
        for i in model.I
        for j in model.J
    }
    
    model.w = Param(model.I, model.J, initialize=w_ij) # accecibility
    model.n = Param(model.I, initialize=n_i)
    model.x = Var(model.J, domain=Binary)
    
    # Expresion to define cumulative linear coverage
    def C(model, i):
        return sum(model.w[i,j]*model.x[j] for j in model.J)
    model.C = Expression(model.I, rule=C)
    
    # Objective function (total coverage)
    def risk(model):
        return sum(model.n[i] * exp(-model.C[i]) for i in model.I)
    model.obj = Objective(rule=risk, sense=minimize)
    
    # budget-constraint (K hospitals)
    model.K = Param(initialize=1, mutable=True)
    
    def cap_rule(model):
        return sum(model.x[j] for j in model.J) <= model.K
    
    model.cap = Constraint(rule=cap_rule)
    
    # solver (create ONCE)
    solver = SolverFactory('couenne',executable=r"C:\Users\jesarauj001\Downloads\couenne\bin\couenne.exe")

    return model, solver


# -----------------------
# SIMULATION 
# -----------------------
def SIMULATION(
    spots,
    candidates,
    T, th,
    K_values=None
):
    """
    Runs Pareto simulations for coverage and risk models.
    Returns all metrics for post-processing plots.
    """

    if K_values is None:
        K_values = list(range(1, 13))

    # COVERAGE MODEL
    model_c, solver_c = coverage_model(spots, candidates, T, th)

    pareto_hosp_c = []
    etas_c = []
    u_ratios_c = []
    times_c = []
    ginis_c = []

    print('-------------------------------------')
    print('---------- COVERAGE MODEL------------')
    print('-------------------------------------')

    for K in K_values:

        start = time.time()

        model_c.K.set_value(K)
        solver_c.solve(model_c)
        cov = [ value(model_c.C[i]) for i in spots.index ]

        n_hosp = sum(int(round(value(model_c.x[j]))) for j in model_c.J)
        eta_c = compute_efficiency(model_c)
        mean_time_c = compute_mean_time(T, spots, model_c, candidates)
        u_ratio_c = compute_unprotected_ratio(model_c, spots)
        gini_c = compute_gini(cov)

        elapsed = time.time() - start

        pareto_hosp_c.append(n_hosp)
        etas_c.append(eta_c)
        times_c.append(mean_time_c)
        u_ratios_c.append(u_ratio_c)
        ginis_c.append(gini_c)

        print(f"K={K} | eta={eta_c:.2f}% | mean time={mean_time_c:.2f}s | ratio={u_ratio_c:.2f}% | duration={elapsed:.2f}s")

    # RISK MODEL
    model_r, solver_r = risk_model(spots, candidates, T, th)

    pareto_hosp_r = []
    etas_r = []
    times_r = []
    u_ratios_r = []
    ginis_r = []

    print('-------------------------------------')
    print('------------- RISK MODEL-------------')
    print('-------------------------------------')
    
    for K in K_values:

        start = time.time()

        model_r.K.set_value(K)
        solver_r.solve(model_r)
        cov = [ value(model_r.C[i]) for i in spots.index ]
        
        n_hosp = sum(int(round(value(model_r.x[j]))) for j in model_r.J)
        eta_r = compute_efficiency(model_r)
        mean_time_r = compute_mean_time(T, spots, model_r, candidates)
        u_ratio_r = compute_unprotected_ratio(model_r, spots)
        gini_r = compute_gini(cov)
        
        elapsed = time.time() - start

        pareto_hosp_r.append(n_hosp)
        etas_r.append(eta_r)
        times_r.append(mean_time_r)
        u_ratios_r.append(u_ratio_r)
        ginis_r.append(gini_r)

        print(f"K={K} | eta={eta_r:.2f}% | mean time={mean_time_r:.2f}s | ratio={u_ratio_r:.2f}% | duration={elapsed:.2f}s")

    # OUTPUT
    return {
        "K_values": K_values,

        "coverage": {
            "hospitals": pareto_hosp_c,
            "eta_c": etas_c,
            "time_c": times_c,
            'ratio_c':u_ratios_c,
            'gini_c':ginis_c            
        },

        "risk": {
            "hospitals": pareto_hosp_r,
            "eta_r": etas_r,
            "time_r": times_r,
            'ratio_r':u_ratios_r,
            'gini_r':ginis_r
        }
    }


## PLOT FUNCTION

# ------------------------------------------------------------------
# COVERAGE / RISK VISUALIZATION
# ------------------------------------------------------------------
def plot_accessibility_results(
    spots,
    model,
    opened_hosp,
    map_center=[46.766429, -71.289947],
    map_zoom=10
):
    """
    Computes accessibility metrics, plots histograms,
    and returns an interactive folium map.
    """

    # -----------------------
    # STEP 1: compute raw coverage
    # -----------------------
    spots = spots.copy()

    spots["C"] = [
        value(model.C[i])
        for i in spots.index
    ]

    # -----------------------
    # STEP 2: residual risk
    # -----------------------
    R = spots["events"] * np.exp(-spots["C"])

    R_norm = round(normalize(R), 2)

    # -----------------------
    # STEP 3: normalize coverage
    # -----------------------
    C_norm = round(normalize(spots["C"]), 2)

    spots["C_norm"] = C_norm

    # -----------------------
    # STEP 4: histograms
    # -----------------------
    fig, axes = plt.subplots(1, 2, figsize=(12,4))

    axes[0].hist(
        C_norm,
        bins=20,
        color="green",
        edgecolor="black"
    )

    axes[0].set_title("Hospital Accessibility")
    #axes[0].set_xlabel("Accessibility")
    axes[0].set_ylabel("Frequency")

    axes[1].hist(
        R_norm,
        bins=20,
        color="red",
        edgecolor="black"
    )

    axes[1].set_title("Residual Vulnerability")
   # axes[1].set_xlabel("Residual Risk")
    axes[1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()

    # -----------------------
    # STEP 5: interactive map
    # -----------------------
    map_quebec = folium.Map(
        location=map_center,
        zoom_start=map_zoom
    )

    # Demand points
    for idx, row in spots.iterrows():

        radius = np.sqrt(row["events"]) * 3

        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],

            radius=radius,

            color=None,
            weight=0,

            fill=True,
            fill_color="crimson",
            fill_opacity=0.7,

            tooltip=(
                f"<b>ID:</b> {row.name}<br>"
                f"Events: {row['events']}<br>"
                f"Age: {row['age']}<br>"
                f"Disease: {row['disease']}<br>"
                f"Weight: {row['weight']}<br>"
                f"Travel time: {round(row['nearest_time'],2)}<br>"
                f"Coverage: {row['C_norm']}"
            )

        ).add_to(map_quebec)

    # Open hospitals
    for idx, row in opened_hosp.iterrows():

        folium.Marker(
            location=[row["latitude"], row["longitude"]],

            icon=folium.Icon(
                color="black",
                icon="plus-sign",
                prefix="glyphicon"
            ),

            tooltip=f"Hospital {idx}"

        ).add_to(map_quebec)

    # Heatmap
    heat_data = [
        [row.geometry.y, row.geometry.x, row["C_norm"]]
        for idx, row in spots.iterrows()
        if row["C_norm"] > 0.05
    ]

    HeatMap(
        heat_data,
        radius=15,
        blur=20
    ).add_to(map_quebec)

    return map_quebec


import numpy as np
import matplotlib.pyplot as plt

def plot_model_comparison(results, figsize=(18, 8)):
    """
    Plot comparison between Model 1 (coverage) and Model 2 (risk)
    across different K values.
    
    Parameters
    ----------
    results : dict
        Dictionary containing:
        - K_values
        - coverage: {eta_c, time_c, ratio_c, gini_c}
        - risk: {eta_r, time_r, ratio_r, gini_r}
    figsize : tuple
        Figure size for matplotlib
    """
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    x = np.array(results["K_values"])
    
    # (1) Efficiency metric
    axes[0,0].plot(x, results["coverage"]["eta_c"], marker='o', label="Model 1")
    axes[0,0].plot(x, results["risk"]["eta_r"], marker='o', label="Model 2")
    axes[0,0].set_title("Efficiency")
    axes[0,0].set_ylabel("%")
    axes[0,0].legend()

    # (2) Mean time metric
    axes[0,1].plot(x, results["coverage"]["time_c"], marker='o', label="Model 1")
    axes[0,1].plot(x, results["risk"]["time_r"], marker='o', label="Model 2")
    axes[0,1].set_title("Mean time")
    axes[0,1].set_ylabel("minutes")
    axes[0,1].legend()

    # (3) Unprotected ratio 
    axes[1,0].plot(x, results["coverage"]["ratio_c"], marker='o', label="Model 1")
    axes[1,0].plot(x, results["risk"]["ratio_r"], marker='o', label="Model 2")
    axes[1,0].set_title("Unprotected ratio")
    axes[1,0].set_ylabel("%")
    axes[1,0].legend()

    # (4) GINI INDEX
    axes[1,1].plot(x, results["coverage"]["gini_c"], marker='o', label="Model 1")
    axes[1,1].plot(x, results["risk"]["gini_r"], marker='o', label="Model 2")
    axes[1,1].set_title("Gini Index")
    axes[1,1].legend()

    plt.tight_layout()
    plt.show()

    return fig, axes


# -----------------------
# Time plots
# -----------------------
def time_plots(time_matrix):
    """
    This function plots the histogram of travel times from patient i to hospital j and the survival curve
    """

    # Flatten and clean
    values = time_matrix.ravel()
    
    # Survival curve
    t = np.sort(values)
    n = len(t)
    survival = 1.0 - np.arange(1, n + 1) / n
    
    # Percentiles
    p50, p90, p95, p99 = np.percentile(values, [50, 90, 95, 99])
    
    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    # Histogram
    ax[0].hist(values, bins=20, color="#4C72B0", edgecolor="black", alpha=0.85)
    
    ax[0].set_title("Histogram of travel times")
    ax[0].set_xlabel("Minutes")
    ax[0].set_ylabel("Frequency")
    ax[0].grid(alpha=0.3)
    
    # Stats box
    stats_text = (
        f"N (pairs): {len(values):,}\n"
        f"Mean: {values.mean():.1f} min\n"
        f"Median: {np.median(values):.1f} min"
        f"Std: {values.std():.1f}\n"
        f"Max: {values.max():.1f} min\n"
    )
    
    ax[0].text(
        0.68, 0.95, stats_text,
        transform=ax[0].transAxes,
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
    )
    
    # Survival curve
    ax[1].plot(t, survival, color="blue", linewidth=2)
    
    ax[1].set_title("Survival curve S(t) = P(T > t)")
    ax[1].set_xlabel("Minutes")
    ax[1].set_ylabel("S(t)")
    ax[1].grid(alpha=0.3)
    
    # Key points
    def S_at(x):
        return np.mean(values > x)
    
    points = [15, 20, 30]
    colors = ["red", "orange", "green"]
    
    for x, c in zip(points, colors):
        y = S_at(x)
        ax[1].scatter(x, y, color=c, s=80, zorder=3)
        ax[1].hlines(y, 0, x, linestyles="dashed", colors=c, alpha=0.7)
        ax[1].vlines(x, 0, y, linestyles="dashed", colors=c, alpha=0.7)
        ax[1].text(x + 0.5, y, f"{y*100:.0f}%", color=c, fontsize=11)
    
    # Percentiles strip (bottom-style annotation)
    percentile_text = (
        f"P50: {p50:.1f}   "
        f"P90: {p90:.1f}   "
        f"P95: {p95:.1f}   "
        f"P99: {p99:.1f}"
    )
    
    fig.text(0.5, -0.02, percentile_text, ha="center", fontsize=11,
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
    
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# First map showing heart attacks' and hospitals' spatial distribution 
# ---------------------------------------------------------------------
def plot_map(
    boundaries,
    spots,
    candidates=None,
    use_clusters=False,
    cluster_col="cluster",
    n_clusters=None
):
    """
    Map of events and optionally candidate hospitals.
    """

    colormap = plt.colormaps["tab10"]

    map_quebec = folium.Map(
        location=[46.766429, -71.289947],
        zoom_start=10
    )

    # -----------------------
    # Boundaries
    # -----------------------
    folium.GeoJson(
        boundaries,
        name="Límites Quebec",
        style_function=lambda x: {
            'fillColor': 'blue',
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.3
        },
        tooltip=folium.GeoJsonTooltip(fields=["district"])
    ).add_to(map_quebec)

    # -----------------------
    # Spots
    # -----------------------
    for idx, row in spots.iterrows():

        rad = np.sqrt(row["events"]) * 3
        color = "crimson"

        tooltip_text = (
            f"Events: {row['events']}<br>"
            f"Age: {row['age']}<br>"
            f"Disease: {row['disease']}<br>"
            f"Weight: {row['weight']}"
        )

        if use_clusters:
            cluster = row.get(cluster_col, None)

            if cluster is not None:
                if cluster == -1:
                    color = "black"
                else:
                    if n_clusters is None:
                        n_clusters = spots[cluster_col].nunique()

                    color = colors.rgb2hex(
                        colormap(cluster % 10)
                    )

                tooltip_text = f"Cluster: {cluster}<br>" + tooltip_text

        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=rad,
            color=color,
            weight=1 if use_clusters else 0,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            tooltip=tooltip_text
        ).add_to(map_quebec)

    # -----------------------
    # Candidates (optional)
    # -----------------------
    if candidates is not None:
        for idx, row in candidates.iterrows():

            folium.Marker(
                location=[row["latitude"], row["longitude"]],
                icon=folium.Icon(
                    color="green",
                    icon="plus-sign",
                    prefix="glyphicon"
                ),
                tooltip=f"Candidate Hospital {idx}"
            ).add_to(map_quebec)

    return map_quebec
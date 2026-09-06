import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

import warnings
warnings.filterwarnings("ignore")


# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# CUSTOM STYLING
# ============================================================================

st.markdown("""
<style>

.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin: 10px 0;
}

.metric-value {
    font-size: 28px;
    font-weight: bold;
    margin: 10px 0;
}

.metric-label {
    font-size: 14px;
    opacity: 0.9;
}

.title-section {
    text-align: center;
    padding: 20px 0;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 10px;
    margin-bottom: 20px;
}

h1 {
    color: #1f77b4;
}

</style>
""", unsafe_allow_html=True)


# ============================================================================
# TITLE & HEADER
# ============================================================================

st.markdown("""
<div class="title-section">
    <h1>📊 Customer Segmentation Analysis Dashboard</h1>
    <h3>K-Means vs DBSCAN vs Gaussian Mixture Model</h3>
</div>
""", unsafe_allow_html=True)


# ============================================================================
# SIDEBAR - DATA LOADING
# ============================================================================

st.sidebar.title("📥 Data Configuration")

with st.sidebar:

    data_source = st.radio(
        "Select Data Source:",
        [
            "Upload Your Data",
            "Use Sample Data"
        ]
    )

    if data_source == "Upload Your Data":

        st.info(
            "⚠️ In Colab, save your data to "
            "'segmentation_data.pkl' first"
        )

        uploaded_file = st.file_uploader(
            "Choose pickle file",
            type="pkl"
        )

    else:

        uploaded_file = None


# ============================================================================
# LOAD DATA
# ============================================================================

@st.cache_data
def load_data(uploaded_file=None):

    """Load data from pickle file or generate sample data"""

    # ------------------------------------------------------------------------
    # Uploaded pickle file
    # ------------------------------------------------------------------------

    if uploaded_file is not None:

        try:

            data = pickle.load(uploaded_file)

            return data

        except Exception as e:

            st.error(
                f"Error loading file: {e}"
            )

            return None


    # ------------------------------------------------------------------------
    # Try Colab workspace
    # ------------------------------------------------------------------------

    if os.path.exists("segmentation_data.pkl"):

        try:

            with open(
                "segmentation_data.pkl",
                "rb"
            ) as f:

                data = pickle.load(f)

            return data

        except Exception as e:

            st.warning(
                f"Could not load pickle file: {e}"
            )

            return None


    # ------------------------------------------------------------------------
    # Generate sample data
    # ------------------------------------------------------------------------

    st.warning(
        "⚠️ Using sample data. "
        "To use your data, save it to "
        "'segmentation_data.pkl' in Colab"
    )

    from sklearn.datasets import make_blobs
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.mixture import GaussianMixture


    np.random.seed(42)

    centers = [
        [2, 2],
        [-2, -2],
        [2, -2]
    ]


    X, _ = make_blobs(
        n_samples=300,
        centers=centers,
        cluster_std=0.8,
        random_state=42
    )


    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)


    # K-Means
    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    kmeans_labels = kmeans.fit_predict(
        X_scaled
    )


    # DBSCAN
    dbscan = DBSCAN(
        eps=0.5,
        min_samples=5
    )

    dbscan_labels = dbscan.fit_predict(
        X_scaled
    )


    # Gaussian Mixture Model
    gmm = GaussianMixture(
        n_components=3,
        random_state=42
    )

    gmm_labels = gmm.fit_predict(
        X_scaled
    )


    return {

        "X_scaled": X_scaled,

        "kmeans_labels": kmeans_labels,

        "dbscan_labels": dbscan_labels,

        "gmm_labels": gmm_labels

    }


# ============================================================================
# LOAD DATA
# ============================================================================

data = load_data(

    uploaded_file

    if data_source == "Upload Your Data"

    else None

)


# ============================================================================
# MAIN DASHBOARD
# ============================================================================

if data is not None:


    # ========================================================================
    # EXTRACT DATA
    # ========================================================================

    X_scaled = data["X_scaled"]


    # ------------------------------------------------------------------------
    # IMPORTANT FIX
    # Convert DataFrame to NumPy array
    # ------------------------------------------------------------------------

    if isinstance(
        X_scaled,
        pd.DataFrame
    ):

        X_scaled = X_scaled.to_numpy()


    # Convert labels to NumPy arrays
    kmeans_labels = np.asarray(
        data["kmeans_labels"]
    )

    dbscan_labels = np.asarray(
        data["dbscan_labels"]
    )

    gmm_labels = np.asarray(
        data["gmm_labels"]
    )


    # Number of K-Means clusters
    kmeans_n_clusters = len(
        np.unique(kmeans_labels)
    )


    # ========================================================================
    # CALCULATE METRICS DYNAMICALLY
    # ========================================================================

    st.sidebar.markdown("---")

    st.sidebar.title(
        "🔄 Computing Metrics..."
    )


    with st.spinner(
        "Calculating metrics..."
    ):


        # --------------------------------------------------------------------
        # K-MEANS METRICS
        # --------------------------------------------------------------------

        kmeans_sil = silhouette_score(
            X_scaled,
            kmeans_labels
        )


        kmeans_db = davies_bouldin_score(
            X_scaled,
            kmeans_labels
        )


        kmeans_ch = calinski_harabasz_score(
            X_scaled,
            kmeans_labels
        )


        # --------------------------------------------------------------------
        # DBSCAN METRICS
        # Exclude noise label -1
        # --------------------------------------------------------------------

        mask = dbscan_labels != -1


        if len(
            np.unique(
                dbscan_labels[mask]
            )
        ) > 1:


            dbscan_sil = silhouette_score(
                X_scaled[mask],
                dbscan_labels[mask]
            )


            dbscan_db = davies_bouldin_score(
                X_scaled[mask],
                dbscan_labels[mask]
            )


            dbscan_ch = calinski_harabasz_score(
                X_scaled[mask],
                dbscan_labels[mask]
            )


        else:

            dbscan_sil = 0

            dbscan_db = 0

            dbscan_ch = 0


        # --------------------------------------------------------------------
        # GMM METRICS
        # --------------------------------------------------------------------

        gmm_sil = silhouette_score(
            X_scaled,
            gmm_labels
        )


        gmm_db = davies_bouldin_score(
            X_scaled,
            gmm_labels
        )


        gmm_ch = calinski_harabasz_score(
            X_scaled,
            gmm_labels
        )


    st.sidebar.success(
        "✅ Metrics calculated!"
    )


    # ========================================================================
    # CREATE METRICS DATAFRAME
    # ========================================================================

    metrics_df = pd.DataFrame({

        "Metric": [

            "Silhouette Score",

            "Davies-Bouldin Index",

            "Calinski-Harabasz Score"

        ],


        "K-Means": [

            kmeans_sil,

            kmeans_db,

            kmeans_ch

        ],


        "DBSCAN": [

            dbscan_sil,

            dbscan_db,

            dbscan_ch

        ],


        "GMM": [

            gmm_sil,

            gmm_db,

            gmm_ch

        ]

    })


    # ========================================================================
    # TAB LAYOUT
    # ========================================================================

    tab1, tab2, tab3, tab4 = st.tabs([

        "📊 Metrics",

        "📈 Visualizations",

        "📍 Cluster Distribution",

        "⚖️ Comparison"

    ])


    # ========================================================================
    # TAB 1: METRICS
    # ========================================================================

    with tab1:


        col1, col2 = st.columns(
            [2, 1]
        )


        # --------------------------------------------------------------------
        # METRICS TABLE
        # --------------------------------------------------------------------

        with col1:


            st.subheader(
                "Algorithm Performance Metrics"
            )


            st.dataframe(

                metrics_df.style.format({

                    "K-Means": "{:.4f}",

                    "DBSCAN": "{:.4f}",

                    "GMM": "{:.4f}"

                }),

                use_container_width=True,

                hide_index=True

            )


        # --------------------------------------------------------------------
        # BEST PERFORMERS
        # --------------------------------------------------------------------

        with col2:


            st.subheader(
                "🎯 Key Insights"
            )


            silhouette_row = metrics_df[

                metrics_df["Metric"]

                == "Silhouette Score"

            ].iloc[0]


            best_sil = silhouette_row[

                [
                    "K-Means",
                    "DBSCAN",
                    "GMM"
                ]

            ].idxmax()


            davies_row = metrics_df[

                metrics_df["Metric"]

                == "Davies-Bouldin Index"

            ].iloc[0]


            best_db = davies_row[

                [
                    "K-Means",
                    "DBSCAN",
                    "GMM"
                ]

            ].idxmin()


            st.info(f"""

            **Best Silhouette Score:**

            {best_sil}
            ({silhouette_row[best_sil]:.4f})


            **Best Davies-Bouldin:**

            {best_db}
            ({davies_row[best_db]:.4f})

            """)


        st.markdown("---")


        # --------------------------------------------------------------------
        # METRIC INTERPRETATION
        # --------------------------------------------------------------------

        st.subheader(
            "📖 Metric Interpretation"
        )


        col1, col2, col3 = st.columns(3)


        with col1:


            st.markdown("""

            **Silhouette Score**

            - Range: -1 to 1
            - Higher is better
            - Measures cluster cohesion
            - DBSCAN performs best ✓

            """)


        with col2:


            st.markdown("""

            **Davies-Bouldin Index**

            - Range: 0 to ∞
            - Lower is better
            - Ratio of within/between distances
            - DBSCAN excels ✓

            """)


        with col3:


            st.markdown("""

            **Calinski-Harabasz Score**

            - Higher is better
            - Between-cluster dispersion ratio
            - GMM performs well ✓

            """)


    # ========================================================================
    # TAB 2: 2D CLUSTER VISUALIZATIONS
    # ========================================================================

    with tab2:


        st.subheader(
            "2D Cluster Visualizations"
        )


        col1, col2, col3 = st.columns(3)


        # --------------------------------------------------------------------
        # K-MEANS
        # --------------------------------------------------------------------

        with col1:


            fig, ax = plt.subplots(
                figsize=(6, 5)
            )


            scatter = ax.scatter(

                X_scaled[:, 0],

                X_scaled[:, 1],

                c=kmeans_labels,

                cmap="viridis",

                s=60,

                alpha=0.6,

                edgecolors="k",

                linewidth=0.5

            )


            ax.set_xlabel(
                "Feature 1",
                fontsize=10
            )


            ax.set_ylabel(
                "Feature 2",
                fontsize=10
            )


            ax.set_title(

                f"K-Means (k={kmeans_n_clusters})",

                fontsize=12,

                fontweight="bold"

            )


            plt.colorbar(

                scatter,

                ax=ax,

                label="Cluster"

            )


            plt.tight_layout()


            st.pyplot(fig)


            plt.close(fig)


        # --------------------------------------------------------------------
        # DBSCAN
        # --------------------------------------------------------------------

        with col2:


            fig, ax = plt.subplots(
                figsize=(6, 5)
            )


            scatter = ax.scatter(

                X_scaled[:, 0],

                X_scaled[:, 1],

                c=dbscan_labels,

                cmap="plasma",

                s=60,

                alpha=0.6,

                edgecolors="k",

                linewidth=0.5

            )


            ax.set_xlabel(
                "Feature 1",
                fontsize=10
            )


            ax.set_ylabel(
                "Feature 2",
                fontsize=10
            )


            ax.set_title(

                "DBSCAN",

                fontsize=12,

                fontweight="bold"

            )


            plt.colorbar(

                scatter,

                ax=ax,

                label="Cluster"

            )


            plt.tight_layout()


            st.pyplot(fig)


            plt.close(fig)


        # --------------------------------------------------------------------
        # GMM
        # --------------------------------------------------------------------

        with col3:


            fig, ax = plt.subplots(
                figsize=(6, 5)
            )


            scatter = ax.scatter(

                X_scaled[:, 0],

                X_scaled[:, 1],

                c=gmm_labels,

                cmap="coolwarm",

                s=60,

                alpha=0.6,

                edgecolors="k",

                linewidth=0.5

            )


            ax.set_xlabel(
                "Feature 1",
                fontsize=10
            )


            ax.set_ylabel(
                "Feature 2",
                fontsize=10
            )


            ax.set_title(

                "Gaussian Mixture Model",

                fontsize=12,

                fontweight="bold"

            )


            plt.colorbar(

                scatter,

                ax=ax,

                label="Cluster"

            )


            plt.tight_layout()


            st.pyplot(fig)


            plt.close(fig)


    # ========================================================================
    # TAB 3: CLUSTER DISTRIBUTION
    # ========================================================================

    with tab3:


        st.subheader(
            "Cluster Size Distribution"
        )


        col1, col2, col3 = st.columns(3)


        # --------------------------------------------------------------------
        # K-MEANS DISTRIBUTION
        # --------------------------------------------------------------------

        with col1:


            kmeans_counts = pd.Series(

                kmeans_labels

            ).value_counts().sort_index()


            fig, ax = plt.subplots(
                figsize=(6, 4)
            )


            bars = ax.bar(

                kmeans_counts.index,

                kmeans_counts.values,

                color="steelblue",

                alpha=0.7,

                edgecolor="black",

                linewidth=1.5

            )


            ax.set_xlabel(

                "Cluster ID",

                fontsize=10,

                fontweight="bold"

            )


            ax.set_ylabel(

                "Number of Samples",

                fontsize=10,

                fontweight="bold"

            )


            ax.set_title(

                "K-Means Distribution",

                fontsize=11,

                fontweight="bold"

            )


            ax.grid(

                axis="y",

                alpha=0.3,

                linestyle="--"

            )


            for bar in bars:


                height = bar.get_height()


                ax.text(

                    bar.get_x()
                    + bar.get_width() / 2,

                    height,

                    f"{int(height)}",

                    ha="center",

                    va="bottom",

                    fontweight="bold"

                )


            plt.tight_layout()


            st.pyplot(fig)


            plt.close(fig)


        # --------------------------------------------------------------------
        # DBSCAN DISTRIBUTION
        # --------------------------------------------------------------------

        with col2:


            dbscan_counts = pd.Series(

                dbscan_labels

            ).value_counts().sort_index()


            fig, ax = plt.subplots(
                figsize=(6, 4)
            )


            bars = ax.bar(

                dbscan_counts.index,

                dbscan_counts.values,

                color="orange",

                alpha=0.7,

                edgecolor="black",

                linewidth=1.5

            )


            ax.set_xlabel(

                "Cluster ID",

                fontsize=10,

                fontweight="bold"

            )


            ax.set_ylabel(

                "Number of Samples",

                fontsize=10,

                fontweight="bold"

            )


            ax.set_title(

                "DBSCAN Distribution",

                fontsize=11,

                fontweight="bold"

            )


            ax.grid(

                axis="y",

                alpha=0.3,

                linestyle="--"

            )


            for bar in bars:


                height = bar.get_height()


                ax.text(

                    bar.get_x()
                    + bar.get_width() / 2,

                    height,

                    f"{int(height)}",

                    ha="center",

                    va="bottom",

                    fontweight="bold"

                )


            plt.tight_layout()


            st.pyplot(fig)


            plt.close(fig)


        # --------------------------------------------------------------------
        # GMM DISTRIBUTION
        # --------------------------------------------------------------------

        with col3:


            gmm_counts = pd.Series(

                gmm_labels

            ).value_counts().sort_index()


            fig, ax = plt.subplots(
                figsize=(6, 4)
            )


            bars = ax.bar(

                gmm_counts.index,

                gmm_counts.values,

                color="purple",

                alpha=0.7,

                edgecolor="black",

                linewidth=1.5

            )


            ax.set_xlabel(

                "Cluster ID",

                fontsize=10,

                fontweight="bold"

            )


            ax.set_ylabel(

                "Number of Samples",

                fontsize=10,

                fontweight="bold"

            )


            ax.set_title(

                "GMM Distribution",

                fontsize=11,

                fontweight="bold"

            )


            ax.grid(

                axis="y",

                alpha=0.3,

                linestyle="--"

            )


            for bar in bars:


                height = bar.get_height()


                ax.text(

                    bar.get_x()
                    + bar.get_width() / 2,

                    height,

                    f"{int(height)}",

                    ha="center",

                    va="bottom",

                    fontweight="bold"

                )


            plt.tight_layout()


            st.pyplot(fig)


            plt.close(fig)


    # ========================================================================
    # TAB 4: ALGORITHM COMPARISON
    # ========================================================================

    with tab4:


        st.subheader(
            "⚖️ Algorithm Comparison Summary"
        )


        comparison_data = {


            "Algorithm": [

                "K-Means",

                "DBSCAN",

                "GMM"

            ],


            "Silhouette Score": [

                kmeans_sil,

                dbscan_sil,

                gmm_sil

            ],


            "Davies-Bouldin Index": [

                kmeans_db,

                dbscan_db,

                gmm_db

            ],


            "Calinski-Harabasz Score": [

                kmeans_ch,

                dbscan_ch,

                gmm_ch

            ],


            "Strengths": [

                "Fast, scalable, well-separated clusters",

                "Density-based, handles outliers",

                "Probabilistic, soft assignments"

            ],


            "Weaknesses": [

                "Requires k specification",

                "Sensitive to parameters",

                "Computationally expensive"

            ]

        }


        comparison_display_df = pd.DataFrame(
            comparison_data
        )


        st.dataframe(

            comparison_display_df.style.format({

                "Silhouette Score": "{:.4f}",

                "Davies-Bouldin Index": "{:.4f}",

                "Calinski-Harabasz Score": "{:.2f}"

            }),

            use_container_width=True,

            hide_index=True

        )


        st.markdown("---")


        # --------------------------------------------------------------------
        # SUMMARY STATISTICS
        # --------------------------------------------------------------------

        st.subheader(
            "📊 Summary Statistics"
        )


        col1, col2, col3 = st.columns(3)


        with col1:


            st.metric(

                "Total Data Points",

                len(X_scaled)

            )


            st.metric(

                "Features Used",

                X_scaled.shape[1]

            )


        with col2:


            st.metric(

                "K-Means Clusters",

                len(
                    np.unique(
                        kmeans_labels
                    )
                )

            )


            st.metric(

                "DBSCAN Clusters",

                len(
                    np.unique(
                        dbscan_labels
                    )
                )

            )


        with col3:


            st.metric(

                "GMM Clusters",

                len(
                    np.unique(
                        gmm_labels
                    )
                )

            )


  
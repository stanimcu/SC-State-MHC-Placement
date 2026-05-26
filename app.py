"""
South Carolina MHC Placement Decision Tool (v8.2)
==================================================
- County-first selection shows county overview with ZIP labels
- Selecting a ZIP zooms into that ZIP
- Running analysis shows results for that ZIP
- Compact, cleaner controls for target and eligible site types
- Selected-site popups show coverage values
- Maximum Coverage Location-Allocation (MCLP)
- Primary action moved to top of sidebar
- Nested target variable categories
- Selected-site star marker matches site type color
- Dynamic result titles based on current selections
- Larger legend styling for proposed sites
- Road network toggle moved outside display settings
- Target selector placed above geographic scope
- County ZIP dropdown ranks 10%+ overlap ZIPs by selected target variable
- County overview map colors and labels 10%+ overlap ZIPs by selected target variable rank
- Legend: squares for site types, clear demand labels, no target variable line
- ZIP dropdown restricted to county ZIPs when county selected
- Default recommended sites set to 3

Author: Tanim
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import folium
import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pulp import LpMaximize, LpProblem, LpVariable, PULP_CBC_CMD, lpSum
from shapely.geometry import Polygon
from streamlit_folium import st_folium

from config import JSON_PATH

warnings.filterwarnings("ignore")

# ===========================
# PAGE CONFIG
# ===========================
st.set_page_config(
    page_title="SC MHC Placement Decision Tool",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===========================
# TARGET VARIABLE OPTIONS
# ===========================
TARGET_VARIABLE_OPTIONS = {
    "Uninsured Population": "uninsured_pop",
    "Total Population": "tot_pop",
    "Disease burden (placeholder)": "tot_hh",
    "Male Adult Population (20+)": "male_adult",
    "Female Adult Population (20+)": "female_adult",
    "Uninsured Under 19": "uninsured_under19",
    "Uninsured 20-34": "uninsured_20_34",
    "Uninsured 35-64": "uninsured_35_64",
    "Uninsured 65+": "uninsured_65plus",
    "Population 0-5": "pop_0_5",
    "Population 0-19": "pop_0_19",
    "Population 20-34": "pop_20_34",
    "Population 35-64": "pop_35_64",
    "Population 65+": "pop_65plus",
    "Non-White Population": "nonwhite_pop",
    "Hispanic Population": "hispanic_pop",
    "Zero-Vehicle Households": "zero_vehicle_hh",
    "Enrolled in School": "enrolled_school",
    "Non-English at Home": "non_english_home",
    "Worker Population": "worker_pop",
    "Veteran Population": "veteran_pop",
}

TARGET_CATEGORIES = {
    "Health insurance": [
        "Uninsured Population",
        "Uninsured Under 19",
        "Uninsured 20-34",
        "Uninsured 35-64",
        "Uninsured 65+",
    ],
    "General population": [
        "Total Population",
        "Population 0-5",
        "Population 0-19",
        "Population 20-34",
        "Population 35-64",
        "Population 65+",
    ],
    "Demographics": [
        "Male Adult Population (20+)",
        "Female Adult Population (20+)",
        "Non-White Population",
        "Hispanic Population",
        "Non-English at Home",
    ],
    "Socio economic": [
        "Zero-Vehicle Households",
        "Enrolled in School",
        "Worker Population",
        "Veteran Population",
    ],
    "Disease burden": [
        "Disease burden (placeholder)",
    ],
}

# ===========================
# FACILITY TYPE COLORS
# ===========================
FACILITY_COLOR_PALETTE = [
    "#800080", "#DC143C", "#228B22", "#4169E1", "#8B0000", "#FF69B4",
    "#FF8C00", "#008080", "#6A5ACD", "#2E8B57", "#DAA520", "#708090",
    "#CD853F", "#4682B4", "#D2691E", "#9370DB", "#3CB371", "#BC8F8F",
    "#5F9EA0", "#E9967A", "#8FBC8F", "#B8860B", "#483D8B", "#2F4F4F",
    "#C71585", "#006400", "#191970", "#8B4513", "#556B2F", "#A0522D",
]

# Demand indicator colors — chosen to not overlap with site type palette
COVERED_COLOR = "#2ECC71"      # green
UNCOVERED_COLOR = "#E74C3C"    # red


def get_type_color_map(facility_types):
    """Build a color map for facility types from the palette."""
    sorted_types = sorted(set(facility_types))
    return {
        t: FACILITY_COLOR_PALETTE[i % len(FACILITY_COLOR_PALETTE)]
        for i, t in enumerate(sorted_types)
    }


def build_target_selector(available_targets):
    """Two-step selector: category first, then target within that category."""
    available_labels = set(available_targets.keys())

    category_map = {
        category: [label for label in labels if label in available_labels]
        for category, labels in TARGET_CATEGORIES.items()
    }
    category_map = {
        category: labels
        for category, labels in category_map.items()
        if labels
    }

    if not category_map:
        st.error("No target variables are available in the loaded dataset.")
        st.stop()

    target_category = st.selectbox(
        "Target category",
        options=list(category_map.keys()),
    )

    target_options = category_map[target_category]

    default_target = (
        "Uninsured Population"
        if "Uninsured Population" in target_options
        else target_options[0]
    )

    target_label = st.selectbox(
        f"{target_category} measure",
        options=target_options,
        index=target_options.index(default_target),
    )

    return target_label, available_targets[target_label], target_category


def build_result_title(selected_zip_display, target_label, time_threshold, travel_mode):
    mode_label = "Drive" if travel_mode == "drive" else "Walk"
    return (
        f"Optimal Sites based on {target_label} "
        f"in {int(time_threshold)}-Min {mode_label}, "
        f"for {selected_zip_display}"
    )


# ===========================
# CSS
# ===========================
def local_css():
    st.markdown(
        """
        <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            [data-testid="stToolbar"] {visibility: hidden;}

            .main {
                background-color: #f8f9fb;
            }

            .stButton>button {
                width: 100%;
                border-radius: 10px;
                height: 3em;
                background-color: #004b98;
                color: white;
                font-weight: 700;
                border: 0;
                box-shadow: 0 2px 8px rgba(0, 75, 152, 0.18);
            }

            .stMultiSelect [data-baseweb="tag"] {
                background-color: #004b98 !important;
                border-radius: 20px !important;
                padding: 5px 10px !important;
                margin: 2px !important;
                color: white !important;
            }

            .instruction-box {
                background-color: #eaf4ff;
                padding: 14px 16px;
                border-radius: 12px;
                border: 1px solid #cfe4ff;
                margin-bottom: 14px;
            }

            .small-note {
                font-size: 0.88rem;
                color: #4d5b6a;
            }

            [data-testid="stMetricValue"] {
                font-size: 22px !important;
            }

            [data-testid="stSidebar"] {
                padding-top: 0.25rem;
            }

            [data-testid="stSidebar"] .stButton > button {
                height: 2.7rem;
                margin-top: 0.15rem;
                margin-bottom: 0.35rem;
            }

            [data-testid="stSidebar"] [data-testid="stExpander"] {
                margin-bottom: 0.35rem;
            }

            [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
                gap: 0.35rem;
            }

            [data-testid="stSidebar"] label {
                margin-bottom: 0.1rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


local_css()

# ===========================
# SAFE MAP RENDERING
# ===========================
def render_folium_map(m: folium.Map, key: str = "map", height: int = 640):
    try:
        st_folium(
            m,
            key=key,
            height=height,
            use_container_width=True,
            returned_objects=[],
        )
    except Exception:
        components.html(
            m.get_root().render(),
            height=height,
            scrolling=True,
        )


# ===========================
# CONSTANTS
# ===========================
DEFAULT_USE_NETWORK = False
DEFAULT_NUM_FACILITIES = 3          # Change 6: default to 3
DEFAULT_SHOW_DEMAND_PREVIEW = False
WALKING_SPEED_KMH = 5.0
ZIP_COUNTY_OVERLAP_THRESHOLD = 0.10

SC_HIGHWAY_SPEEDS_KMH = {
    "motorway": 105,
    "motorway_link": 72,
    "trunk": 89,
    "trunk_link": 64,
    "primary": 72,
    "primary_link": 56,
    "secondary": 56,
    "secondary_link": 48,
    "tertiary": 48,
    "tertiary_link": 40,
    "residential": 40,
    "living_street": 24,
    "service": 24,
    "unclassified": 40,
    "road": 40,
}
SC_FALLBACK_SPEED_KMH = 40
TURN_PENALTY_SECONDS = 5
CIRCUITY_FACTOR = 1.20
DEFAULT_DRIVING_SPEED = 25
MAX_SNAP_DIST_M = 2000

# ===========================
# SESSION STATE
# ===========================
for skey, default in [
    ("analysis_complete", False),
    ("selected_facilities", None),
    ("coverage_matrix", None),
    ("demand_reset", None),
    ("candidates_reset", None),
    ("covered_pop", 0.0),
    ("covered_mask", None),
    ("method_used", "Manhattan Distance"),
    ("selected_cand_ids", None),
    ("covered_dem_ids", None),
    ("target_variable", "uninsured_pop"),
    ("target_label", "Uninsured Population"),
    ("analysis_title", None),
    ("prev_county", None),
    ("prev_zip", None),
    ("view_mode", "county"),
    ("site_metrics_lookup", {}),
]:
    if skey not in st.session_state:
        st.session_state[skey] = default


# ===========================
# HELPERS
# ===========================
def reset_analysis_state():
    for key, value in {
        "analysis_complete": False,
        "selected_facilities": None,
        "coverage_matrix": None,
        "demand_reset": None,
        "candidates_reset": None,
        "covered_pop": 0.0,
        "covered_mask": None,
        "method_used": "Manhattan Distance",
        "selected_cand_ids": None,
        "covered_dem_ids": None,
        "site_metrics_lookup": {},
        "analysis_title": None,
    }.items():
        st.session_state[key] = value


def get_county_union_geometry(county_gdf, county_fips):
    """Return one combined geometry for the selected county."""
    if county_gdf is None or county_fips is None:
        return None

    county_rows = county_gdf[
        county_gdf["COUNTY_FIPS"].astype(str) == str(county_fips)
    ]

    if len(county_rows) == 0:
        return None

    try:
        return county_rows.geometry.union_all()
    except Exception:
        try:
            return county_rows.geometry.unary_union
        except Exception:
            return None


def calculate_zip_overlap_pcts(zip_geometries, county_geom):
    """Calculate ZIP-county overlap percentage (intersection area / ZIP area)."""
    pcts = []

    for geom in zip_geometries:
        try:
            if geom is None or geom.is_empty:
                pcts.append(0.0)
            elif county_geom is None or county_geom.is_empty:
                pcts.append(0.0)
            elif geom.area <= 0:
                pcts.append(0.0)
            else:
                pct = geom.intersection(county_geom).area / geom.area
                pcts.append(float(pct) if np.isfinite(pct) else 0.0)
        except Exception:
            pcts.append(0.0)

    return np.asarray(pcts, dtype=float)


def get_zips_in_county(
    zip_gdf,
    zip_county_map,
    county_fips,
    county_gdf=None,
    min_overlap_pct=ZIP_COUNTY_OVERLAP_THRESHOLD,
):
    if county_gdf is not None:
        county_geom = get_county_union_geometry(county_gdf, county_fips)
        if county_geom is not None:
            pcts = calculate_zip_overlap_pcts(zip_gdf["geometry"], county_geom)
            return zip_gdf[pcts >= min_overlap_pct].copy()

    if zip_county_map is not None and len(zip_county_map) > 0:
        zip_codes = zip_county_map[
            zip_county_map["COUNTY_FIPS"].astype(str) == str(county_fips)
        ]["ZIP_CODE"].astype(str).str.zfill(5).unique()
        return zip_gdf[
            zip_gdf["ZIP_CODE"].astype(str).str.zfill(5).isin(zip_codes)
        ].copy()

    return zip_gdf[zip_gdf["COUNTY_FIPS"].astype(str) == str(county_fips)].copy()


def build_zip_display(zip_row):
    po_name = str(zip_row.get("po_name", "")).strip()
    zip_code = str(zip_row.get("ZIP_CODE", "")).zfill(5)
    return f"{zip_code} ({po_name})" if po_name else zip_code


def get_ordered_zip_choices(
    zip_gdf,
    selected_county_fips=None,
    county_gdf=None,
    demand_df=None,
    target_var=None,
):
    """
    Build ZIP dropdown choices.

    Change 5: When a county is selected, show ONLY ZIPs with 10%+ area overlap
    (no full SC list appended). When no county selected, show all SC ZIPs.
    County ZIPs are ranked by selected target variable, highest first.
    """
    zip_choices = (
        zip_gdf[["ZIP_CODE", "po_name", "geometry"]]
        .drop_duplicates(subset=["ZIP_CODE"])
        .copy()
    )
    zip_choices["ZIP_CODE"] = zip_choices["ZIP_CODE"].astype(str).str.zfill(5)

    if selected_county_fips is not None and county_gdf is not None:
        county_geom = get_county_union_geometry(county_gdf, selected_county_fips)

        if county_geom is None:
            zip_choices = zip_choices.sort_values("ZIP_CODE").reset_index(drop=True)
            zip_choices["zip_label"] = zip_choices.apply(build_zip_display, axis=1)
            return zip_choices.drop(columns=["geometry"], errors="ignore")

        zip_choices["_pct"] = calculate_zip_overlap_pcts(
            zip_choices["geometry"], county_geom
        )

        # Restrict to county ZIPs only
        county_zips = zip_choices[
            zip_choices["_pct"] >= ZIP_COUNTY_OVERLAP_THRESHOLD
        ].copy()

        # Rank by target variable sum
        if (
            demand_df is not None
            and target_var is not None
            and target_var in demand_df.columns
            and "zip_join" in demand_df.columns
            and demand_df["zip_join"].notna().any()
        ):
            demand_tmp = (
                demand_df[["zip_join", target_var]]
                .dropna(subset=["zip_join"])
                .copy()
            )
            demand_tmp["zip_join"] = demand_tmp["zip_join"].astype(str).str.zfill(5)
            demand_tmp[target_var] = pd.to_numeric(
                demand_tmp[target_var], errors="coerce"
            ).fillna(0)

            zip_target_sum = (
                demand_tmp.groupby("zip_join", as_index=False)[target_var]
                .sum()
                .rename(columns={"zip_join": "ZIP_CODE", target_var: "_target_sum"})
            )

            county_zips = county_zips.merge(zip_target_sum, on="ZIP_CODE", how="left")
            county_zips["_target_sum"] = county_zips["_target_sum"].fillna(0.0)
            county_zips = (
                county_zips
                .sort_values(["_target_sum", "ZIP_CODE"], ascending=[False, True])
                .reset_index(drop=True)
            )
            county_zips["_rank"] = range(1, len(county_zips) + 1)

            def build_zip_display_ranked(row):
                po_name = str(row.get("po_name", "")).strip()
                zip_code = str(row.get("ZIP_CODE", "")).zfill(5)
                rank = int(row["_rank"])
                name_part = f" ({po_name})" if po_name else ""
                return f"#{rank} {zip_code}{name_part}"

            county_zips["zip_label"] = county_zips.apply(build_zip_display_ranked, axis=1)
            county_zips = county_zips.drop(
                columns=["_target_sum", "_rank", "_pct", "geometry"], errors="ignore"
            )
        else:
            county_zips = county_zips.sort_values("ZIP_CODE").reset_index(drop=True)
            county_zips["zip_label"] = county_zips.apply(build_zip_display, axis=1)
            county_zips = county_zips.drop(columns=["_pct", "geometry"], errors="ignore")

        return county_zips

    else:
        zip_choices = zip_choices.drop(columns=["geometry"], errors="ignore")
        zip_choices = zip_choices.sort_values("ZIP_CODE").reset_index(drop=True)
        zip_choices["zip_label"] = zip_choices.apply(build_zip_display, axis=1)
        return zip_choices


def get_candidates_in_zip(candidates_df, selected_zip, zip_geom):
    if "zip_join" in candidates_df.columns and candidates_df["zip_join"].notna().any():
        return candidates_df[candidates_df["zip_join"] == selected_zip].copy()
    return candidates_df[candidates_df.geometry.intersects(zip_geom)].copy()


def get_demand_in_zip(demand_df, selected_zip, zip_geom):
    if "zip_join" in demand_df.columns and demand_df["zip_join"].notna().any():
        return demand_df[demand_df["zip_join"] == selected_zip].copy()
    return demand_df[demand_df.geometry.intersects(zip_geom)].copy()


def build_type_rows_html(types_in_zip, type_colors):
    """Change 3: Use colored squares instead of circles for site types."""
    rows = []
    for ftype in types_in_zip:
        fcolor = type_colors.get(ftype, "#808080")
        rows.append(
            f'<p style="margin:2px 0 2px 10px; display:flex; align-items:center;">'
            f'<span style="display:inline-block; width:12px; height:12px; '
            f'background-color:{fcolor}; border-radius:2px; margin-right:6px; '
            f'flex-shrink:0;"></span>'
            f'{ftype}</p>'
        )
    return "\n".join(rows)


def compute_site_metrics(selected_indices, coverage_matrix, demand_weights, candidates_reset):
    site_metrics = {}
    if len(selected_indices) == 0:
        return site_metrics

    demand_weights = np.asarray(demand_weights, dtype=float)

    for row_idx in selected_indices:
        own_mask = coverage_matrix[row_idx, :].astype(bool)
        covered_value = float(demand_weights[own_mask].sum())
        cand_idx = int(candidates_reset.iloc[row_idx]["cand_idx"])
        site_metrics[cand_idx] = {"covered_value": covered_value}

    return site_metrics


# ===========================
# DATA LOADING
# ===========================
@st.cache_data
def load_data(json_path: Path):
    with open(json_path, "r") as f:
        data = json.load(f)

    county_data = data.get("counties", {})
    county_gdf = None

    if county_data:
        geometries, properties = [], []
        for county_id, county_info in county_data.items():
            if "coords" not in county_info:
                continue
            try:
                coords_lonlat = [[pt[1], pt[0]] for pt in county_info["coords"]]
                geom = Polygon(coords_lonlat)
                geometries.append(geom)
                properties.append({
                    "COUNTY_FIPS": str(county_id),
                    "county_name": county_info.get("name", str(county_id)),
                })
            except Exception:
                continue
        if geometries:
            county_gdf = gpd.GeoDataFrame(properties, geometry=geometries, crs="EPSG:4326")

    zip_data = data.get("zip_boundaries", data.get("zips", {}))
    if not zip_data:
        raise ValueError("No ZIP boundaries found in JSON.")

    geometries, properties = [], []
    for zip_code, zip_info in zip_data.items():
        if "coords" not in zip_info:
            continue
        try:
            coords_lonlat = [[pt[1], pt[0]] for pt in zip_info["coords"]]
            geom = Polygon(coords_lonlat)
            geometries.append(geom)
            properties.append({
                "ZIP_CODE": str(zip_code).zfill(5),
                "po_name": zip_info.get("po_name", str(zip_code)),
            })
        except Exception:
            continue

    if not geometries:
        raise ValueError("No valid ZIP geometries.")

    zip_gdf = gpd.GeoDataFrame(properties, geometry=geometries, crs="EPSG:4326")

    candidates_data = data.get("candidate_facilities", data.get("facilities", []))
    if not candidates_data:
        raise ValueError("No candidate facilities found.")
    candidates_df = pd.DataFrame(candidates_data)

    demand_data = data.get("demand_points", data.get("demand", []))
    if not demand_data:
        raise ValueError("No demand points found.")
    demand_df = pd.DataFrame(demand_data)

    candidates_df["cand_idx"] = np.arange(len(candidates_df), dtype=int)
    demand_df["dem_idx"] = np.arange(len(demand_df), dtype=int)

    for df in (candidates_df, demand_df):
        if "zip_code" in df.columns:
            df["zip_code"] = df["zip_code"].astype(str).str.zfill(5)
        for col in ("latitude", "longitude"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    for var_col in TARGET_VARIABLE_OPTIONS.values():
        if var_col in demand_df.columns:
            demand_df[var_col] = pd.to_numeric(demand_df[var_col], errors="coerce").fillna(0)

    candidates_gdf = gpd.GeoDataFrame(
        candidates_df,
        geometry=gpd.points_from_xy(candidates_df["longitude"], candidates_df["latitude"]),
        crs="EPSG:4326",
    )
    demand_gdf = gpd.GeoDataFrame(
        demand_df,
        geometry=gpd.points_from_xy(demand_df["longitude"], demand_df["latitude"]),
        crs="EPSG:4326",
    )

    try:
        _zip = zip_gdf[["ZIP_CODE", "geometry"]].copy()
        candidates_gdf = (
            gpd.sjoin(candidates_gdf, _zip, how="left", predicate="intersects")
            .drop(columns=["index_right"])
        )
        demand_gdf = (
            gpd.sjoin(demand_gdf, _zip, how="left", predicate="intersects")
            .drop(columns=["index_right"])
        )
        candidates_gdf = candidates_gdf.rename(columns={"ZIP_CODE": "zip_join"})
        demand_gdf = demand_gdf.rename(columns={"ZIP_CODE": "zip_join"})
    except Exception:
        if "zip_join" not in candidates_gdf.columns:
            candidates_gdf["zip_join"] = np.nan
        if "zip_join" not in demand_gdf.columns:
            demand_gdf["zip_join"] = np.nan

    zip_county_map = pd.DataFrame(columns=["ZIP_CODE", "COUNTY_FIPS", "county_name"])

    if county_gdf is not None:
        try:
            zip_county_join = (
                gpd.sjoin(
                    zip_gdf[["ZIP_CODE", "po_name", "geometry"]],
                    county_gdf[["COUNTY_FIPS", "county_name", "geometry"]],
                    how="left",
                    predicate="intersects",
                )
                .drop(columns=["index_right"])
            )
            zip_county_map = zip_county_join[
                ["ZIP_CODE", "COUNTY_FIPS", "county_name"]
            ].drop_duplicates()

            zip_centroids = zip_gdf.copy()
            zip_centroids["geometry"] = zip_centroids.geometry.centroid
            zip_primary = (
                gpd.sjoin(
                    zip_centroids[["ZIP_CODE", "geometry"]],
                    county_gdf[["COUNTY_FIPS", "county_name", "geometry"]],
                    how="left",
                    predicate="within",
                )
                .drop(columns=["index_right"])
            )
            zip_gdf["COUNTY_FIPS"] = zip_primary["COUNTY_FIPS"].values
            zip_gdf["county_name"] = zip_primary["county_name"].values
        except Exception:
            zip_gdf["COUNTY_FIPS"] = np.nan
            zip_gdf["county_name"] = np.nan

    all_types = candidates_gdf["type"].dropna().unique().tolist()
    global_type_colors = get_type_color_map(all_types)

    return (
        zip_gdf,
        candidates_gdf,
        demand_gdf,
        county_gdf,
        zip_county_map,
        global_type_colors,
    )


# ===========================
# COUNTY OVERVIEW MAP
# ===========================
def create_county_overview_map(
    county_gdf,
    zip_gdf,
    zip_county_map,
    selected_county_fips,
    tiles="CartoDB positron",
    demand_df=None,
    target_var=None,
    target_label=None,
):
    county_rows = county_gdf[
        county_gdf["COUNTY_FIPS"].astype(str) == str(selected_county_fips)
    ]
    if county_rows.empty:
        raise ValueError("Selected county not found.")

    county_row = county_rows.iloc[0]
    bounds = county_row.geometry.bounds
    center_lat = float((bounds[1] + bounds[3]) / 2)
    center_lon = float((bounds[0] + bounds[2]) / 2)

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles=tiles,
        prefer_canvas=True,
    )

    folium.GeoJson(
        county_gdf.to_json(),
        style_function=lambda _: {
            "fillColor": "transparent",
            "color": "#cccccc",
            "weight": 1,
            "fillOpacity": 0,
            "opacity": 0.4,
        },
    ).add_to(m)

    folium.GeoJson(
        county_row.geometry.__geo_interface__,
        style_function=lambda _: {
            "fillColor": "transparent",
            "color": "#004b98",
            "weight": 4,
            "fillOpacity": 0,
            "opacity": 0.9,
        },
    ).add_to(m)

    county_geom = county_row.geometry
    try:
        pcts = calculate_zip_overlap_pcts(zip_gdf["geometry"], county_geom)
        zips_in_county = zip_gdf[pcts >= ZIP_COUNTY_OVERLAP_THRESHOLD].copy()
    except Exception:
        zips_in_county = zip_gdf.iloc[0:0].copy()

    zip_target_sum = {}
    if (
        demand_df is not None
        and target_var is not None
        and target_var in demand_df.columns
        and "zip_join" in demand_df.columns
        and demand_df["zip_join"].notna().any()
    ):
        demand_tmp = (
            demand_df[["zip_join", target_var]]
            .dropna(subset=["zip_join"])
            .copy()
        )
        demand_tmp["zip_join"] = demand_tmp["zip_join"].astype(str).str.zfill(5)
        demand_tmp[target_var] = pd.to_numeric(
            demand_tmp[target_var], errors="coerce"
        ).fillna(0)
        sums = demand_tmp.groupby("zip_join")[target_var].sum()
        zip_target_sum = {str(k).zfill(5): v for k, v in sums.items()}

    zips_in_county["_target_sum"] = (
        zips_in_county["ZIP_CODE"]
        .astype(str).str.zfill(5)
        .map(zip_target_sum)
        .fillna(0.0)
    )

    zips_in_county = (
        zips_in_county
        .sort_values(["_target_sum", "ZIP_CODE"], ascending=[False, True])
        .reset_index(drop=True)
    )
    zips_in_county["_rank"] = range(1, len(zips_in_county) + 1)
    n_zips = len(zips_in_county)

    def rank_to_color(rank, n):
        if n <= 1:
            return "#1a4f8a"
        t = (rank - 1) / (n - 1)
        r = int(26 + t * (210 - 26))
        g = int(79 + t * (233 - 79))
        b = int(138 + t * (255 - 138))
        return f"#{r:02x}{g:02x}{b:02x}"

    for _, zrow in zips_in_county.iterrows():
        geom = zrow.geometry
        if geom is None or geom.is_empty:
            continue

        rank = int(zrow["_rank"])
        target_val = float(zrow["_target_sum"])
        fill_color = rank_to_color(rank, n_zips)
        label_text = target_label if target_label else target_var

        feature = {
            "type": "Feature",
            "geometry": geom.__geo_interface__,
            "properties": {
                "ZIP_CODE": str(zrow["ZIP_CODE"]),
                "po_name": str(zrow.get("po_name", "")),
                "rank": rank,
                "target_val": int(round(target_val)),
            },
        }

        folium.GeoJson(
            feature,
            style_function=lambda _, fc=fill_color: {
                "fillColor": fc,
                "color": "#1E90FF",
                "weight": 2,
                "fillOpacity": 0.55,
                "opacity": 0.8,
            },
            highlight_function=lambda _: {
                "fillOpacity": 0.85,
                "weight": 4,
                "color": "#004b98",
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["ZIP_CODE", "po_name", "rank", "target_val"],
                aliases=["ZIP:", "Name:", "Rank:", f"{label_text}:"],
                localize=True,
            ),
        ).add_to(m)

        centroid = geom.centroid
        folium.Marker(
            location=[centroid.y, centroid.x],
            icon=folium.DivIcon(
                html=(
                    f'<div style="font-size:11px; font-weight:bold; color:#003366; '
                    f'text-shadow: 1px 1px 2px white, -1px -1px 2px white, '
                    f'1px -1px 2px white, -1px 1px 2px white; white-space:nowrap;">'
                    f'#{rank} {zrow["ZIP_CODE"]}</div>'
                ),
                icon_size=(70, 15),
                icon_anchor=(35, 7),
            ),
        ).add_to(m)

    if n_zips > 0:
        legend_html = f"""
        <div style="
            position: fixed; bottom: 40px; right: 40px; width: 200px;
            background-color: white; z-index:9999; font-size: 13px;
            border:1px solid #c7cfdb; border-radius: 10px; padding: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.12);
        ">
            <p style="margin:0 0 8px 0; font-weight:bold; font-size:14px;">Need Ranking</p>
            <p style="margin:0 0 4px 0; color:#666; font-size:12px;">by {target_label or ""}</p>
            <div style="display:flex; align-items:center; margin-top:8px;">
                <div style="width:100%; height:16px; background: linear-gradient(to right, #1a4f8a, #d2e9ff); border-radius:4px;"></div>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:11px; margin-top:3px;">
                <span>High need</span>
                <span>Low need</span>
            </div>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

    minx, miny, maxx, maxy = map(float, bounds)
    pad_x = max((maxx - minx) * 0.08, 0.01)
    pad_y = max((maxy - miny) * 0.08, 0.01)
    m.fit_bounds([[miny - pad_y, minx - pad_x], [maxy + pad_y, maxx + pad_x]])
    return m


# ===========================
# ZIP-LEVEL MAP
# ===========================
def create_map(
    zip_gdf,
    selected_zip,
    candidates_df,
    demand_df,
    type_colors,
    target_var="uninsured_pop",
    target_label="Uninsured Population",
    selected_cand_ids=None,
    covered_dem_ids=None,
    site_metrics_lookup=None,
    show_demand_preview=False,
    selected_types=None,
    tiles="CartoDB positron",
    county_gdf=None,
):
    zip_boundary = zip_gdf[zip_gdf["ZIP_CODE"] == selected_zip].iloc[0]
    bounds = zip_boundary.geometry.bounds
    center_lat = float((bounds[1] + bounds[3]) / 2)
    center_lon = float((bounds[0] + bounds[2]) / 2)

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles=tiles,
        prefer_canvas=True,
    )

    if county_gdf is not None:
        folium.GeoJson(
            county_gdf.to_json(),
            style_function=lambda _: {
                "fillColor": "transparent",
                "color": "#999999",
                "weight": 1.5,
                "fillOpacity": 0,
                "opacity": 0.5,
            },
        ).add_to(m)

    folium.GeoJson(
        zip_boundary.geometry.__geo_interface__,
        style_function=lambda _: {
            "fillColor": "#E6F2FF",
            "color": "#1E90FF",
            "weight": 4,
            "fillOpacity": 0.08,
            "opacity": 0.9,
        },
    ).add_to(m)

    candidates_in_zip = get_candidates_in_zip(
        candidates_df, selected_zip, zip_boundary.geometry
    )
    if selected_types:
        candidates_in_zip = candidates_in_zip[
            candidates_in_zip["type"].isin(selected_types)
        ].copy()

    demand_in_zip = get_demand_in_zip(demand_df, selected_zip, zip_boundary.geometry)
    types_in_zip = sorted(candidates_in_zip["type"].dropna().unique())
    analysis_complete = selected_cand_ids is not None and covered_dem_ids is not None

    if analysis_complete:
        for _, fac in candidates_in_zip.iterrows():
            lat, lon = float(fac["latitude"]), float(fac["longitude"])
            name = str(fac.get("name", ""))
            ftype = str(fac.get("type", ""))
            cand_idx = int(fac.get("cand_idx", -1))
            is_selected = cand_idx in selected_cand_ids

            if is_selected:
                metrics = (site_metrics_lookup or {}).get(cand_idx, {})
                covered_value = float(metrics.get("covered_value", 0.0))
                star_color = type_colors.get(ftype, "#808080")

                popup_html = f"""
                <div style="width: 240px; font-family: Arial, sans-serif; font-size: 13px; line-height: 1.35;">
                    <div style="font-weight: 700; color: #1f2d3d; margin-bottom: 4px;">Proposed Site</div>
                    <div style="font-weight: 700; color: #004b98; margin-bottom: 2px;">{name}</div>
                    <div style="color: #55606e; margin-bottom: 8px;">{ftype}</div>
                    <table style="width: 100%; border-collapse: collapse;">
                        <tr>
                            <td style="padding: 3px 0; color: #55606e; vertical-align: top;">Covered {target_label}</td>
                            <td style="padding: 3px 0; text-align: right; font-weight: 700; color: #1f2d3d;">{covered_value:,.0f}</td>
                        </tr>
                    </table>
                </div>
                """

                folium.Marker(
                    location=[lat, lon],
                    popup=folium.Popup(popup_html, max_width=280),
                    icon=folium.DivIcon(
                        html=(
                            f'<div style="font-size:36px; line-height:36px; color:{star_color}; '
                            f'text-shadow:0 0 4px rgba(255,255,255,0.95), 0 0 1px rgba(0,0,0,0.35);">'
                            f'&#9733;</div>'
                        ),
                        icon_size=(36, 36),
                        icon_anchor=(18, 18),
                    ),
                ).add_to(m)
            else:
                color = type_colors.get(ftype, "#808080")
                folium.Marker(
                    location=[lat, lon],
                    popup=f"<b>{name}</b><br>{ftype}",
                    icon=folium.DivIcon(
                        html=(
                            f'<div style="width:12px; height:12px; '
                            f'background-color:{color}; border-radius:2px; '
                            f'border:1.5px solid rgba(0,0,0,0.3);"></div>'
                        ),
                        icon_size=(12, 12),
                        icon_anchor=(6, 6),
                    ),
                ).add_to(m)
    else:
        for _, fac in candidates_in_zip.iterrows():
            ftype = str(fac.get("type", ""))
            color = type_colors.get(ftype, "#808080")
            folium.Marker(
                location=[float(fac["latitude"]), float(fac["longitude"])],
                popup=f"<b>{fac.get('name', '')}</b><br>{ftype}",
                icon=folium.DivIcon(
                    html=(
                        f'<div style="width:12px; height:12px; '
                        f'background-color:{color}; border-radius:2px; '
                        f'border:1.5px solid rgba(0,0,0,0.3);"></div>'
                    ),
                    icon_size=(12, 12),
                    icon_anchor=(6, 6),
                ),
            ).add_to(m)

    if analysis_complete or show_demand_preview:
        for _, dem in demand_in_zip.iterrows():
            lat, lon = float(dem["latitude"]), float(dem["longitude"])
            target_val = (
                float(dem[target_var])
                if pd.notna(dem.get(target_var, np.nan))
                else 0.0
            )

            if analysis_complete:
                is_covered = int(dem.get("dem_idx", -1)) in covered_dem_ids
                if is_covered:
                    # Change 2: use COVERED_COLOR, updated popup text
                    color, fill_color = COVERED_COLOR, COVERED_COLOR
                    popup_text = f"<b>Within travel time</b><br>{target_label}: {target_val:,.1f}"
                else:
                    # Change 2: use UNCOVERED_COLOR, updated popup text
                    color, fill_color = UNCOVERED_COLOR, UNCOVERED_COLOR
                    popup_text = f"<b>Outside travel time</b><br>{target_label}: {target_val:,.1f}"
            else:
                color, fill_color = UNCOVERED_COLOR, UNCOVERED_COLOR
                popup_text = f"<b>Census block centroid</b><br>{target_label}: {target_val:,.1f}"

            folium.CircleMarker(
                location=[lat, lon],
                radius=3,
                popup=popup_text,
                color=color,
                fill=True,
                fillColor=fill_color,
                fillOpacity=0.7,
                weight=1,
            ).add_to(m)

    # Changes 1-4: updated legend
    type_rows = build_type_rows_html(types_in_zip, type_colors)

    if analysis_complete:
        legend_html = f"""
        <div style="
            position: fixed; bottom: 40px; right: 40px; width: 300px;
            max-height: 460px; overflow-y: auto; background-color: white; z-index:9999;
            font-size: 14px; border:1px solid #c7cfdb; border-radius: 10px; padding: 14px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.12);
        ">
            <p style="margin:0 0 8px 0; font-weight:bold; font-size:15px;">Map Legend</p>
            <p style="margin:3px 0; font-size:14px;">
                <span style="color:#555; font-size:18px;">&#9733;</span>
                &nbsp;Proposed site (color matches site type)
            </p>
            <p style="margin:3px 0; font-size:14px;">
                <span style="color:{COVERED_COLOR}; font-size:16px;">&#9679;</span>
                &nbsp;Census blocks within travel time
            </p>
            <p style="margin:3px 0; font-size:14px;">
                <span style="color:{UNCOVERED_COLOR}; font-size:16px;">&#9679;</span>
                &nbsp;Census blocks outside travel time
            </p>
            <details style="margin-top:10px;">
                <summary style="cursor:pointer; font-size:14px;"><b>Candidate sites by type ({len(types_in_zip)})</b></summary>
                {type_rows}
            </details>
        </div>
        """
    else:
        if show_demand_preview:
            demand_line = (
                f'<p style="margin:3px 0; font-size:14px;">'
                f'<span style="color:{UNCOVERED_COLOR}; font-size:16px;">&#9679;</span>'
                f'&nbsp;Census block centroids</p>'
            )
        else:
            demand_line = '<p style="margin:3px 0; color:#617184; font-size:14px;">Demand preview hidden</p>'

        legend_html = f"""
        <div style="
            position: fixed; bottom: 40px; right: 40px; width: 300px;
            max-height: 460px; overflow-y: auto; background-color: white; z-index:9999;
            font-size: 14px; border:1px solid #c7cfdb; border-radius: 10px; padding: 14px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.12);
        ">
            <p style="margin:0 0 8px 0; font-weight:bold; font-size:15px;">Map Legend</p>
            {demand_line}
            <details style="margin-top:10px;" open>
                <summary style="cursor:pointer; font-size:14px;"><b>Candidate sites by type ({len(types_in_zip)})</b></summary>
                {type_rows}
            </details>
        </div>
        """

    m.get_root().html.add_child(folium.Element(legend_html))

    minx, miny, maxx, maxy = map(float, bounds)
    pad_x = min(max((maxx - minx) * 0.08, 0.0015), 0.25)
    pad_y = min(max((maxy - miny) * 0.08, 0.0015), 0.25)
    m.fit_bounds([[miny - pad_y, minx - pad_x], [maxy + pad_y, maxx + pad_x]])
    return m


# ===========================
# NETWORK / COVERAGE / SOLVER
# ===========================
def snap_points_to_nodes(G, lons, lats, max_snap_dist_m=MAX_SNAP_DIST_M):
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    out = np.empty(len(lons), dtype=object)

    try:
        nodes, dists = ox.distance.nearest_nodes(G, X=lons, Y=lats, return_dist=True)
        nodes = np.asarray(nodes, dtype=object)
        dists = np.asarray(dists, dtype=float)
        nodes[dists > max_snap_dist_m] = None
        out[:] = nodes
        return out
    except Exception:
        for i, (lo, la) in enumerate(zip(lons, lats)):
            try:
                n, d = ox.distance.nearest_nodes(G, lo, la, return_dist=True)
                out[i] = n if d <= max_snap_dist_m else None
            except Exception:
                out[i] = None
        return out


def estimate_required_graph_dist_m(
    center_lat, center_lon, candidates_df, demand_df, min_dist=15000, buffer_m=5000
):
    pts = []
    if candidates_df is not None and len(candidates_df) > 0:
        pts.append(candidates_df[["latitude", "longitude"]])
    if demand_df is not None and len(demand_df) > 0:
        pts.append(demand_df[["latitude", "longitude"]])
    if not pts:
        return int(min_dist)

    allpts = pd.concat(pts, ignore_index=True).dropna()
    if allpts.empty:
        return int(min_dist)

    try:
        d = ox.distance.great_circle_vec(
            center_lat, center_lon, allpts["latitude"].values, allpts["longitude"].values
        )
        maxd = float(np.nanmax(d))
    except Exception:
        lat1, lon1 = np.radians(center_lat), np.radians(center_lon)
        lat2 = np.radians(allpts["latitude"].values.astype(float))
        lon2 = np.radians(allpts["longitude"].values.astype(float))
        a = (
            np.sin((lat2 - lat1) / 2) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
        )
        maxd = float(np.nanmax(6371000 * 2 * np.arcsin(np.sqrt(a))))

    return int(max(min_dist, maxd + buffer_m)) if np.isfinite(maxd) else int(min_dist)


def preprocess_network_speeds(G, network_type="drive"):
    if network_type == "walk":
        for _, _, _, data in G.edges(data=True, keys=True):
            data["speed_kph"] = WALKING_SPEED_KMH
            data["travel_time"] = (
                (data["length"] / 1000.0 / WALKING_SPEED_KMH) * 3600.0 + TURN_PENALTY_SECONDS
            ) / 60.0
    else:
        try:
            G = ox.add_edge_speeds(G, hwy_speeds=SC_HIGHWAY_SPEEDS_KMH, fallback=SC_FALLBACK_SPEED_KMH)
            G = ox.add_edge_travel_times(G)
        except Exception:
            try:
                G = ox.routing.add_edge_speeds(G, hwy_speeds=SC_HIGHWAY_SPEEDS_KMH, fallback=SC_FALLBACK_SPEED_KMH)
                G = ox.routing.add_edge_travel_times(G)
            except Exception:
                for _, _, _, data in G.edges(data=True, keys=True):
                    rt = data.get("highway", "unclassified")
                    if isinstance(rt, list):
                        rt = rt[0]
                    data["speed_kph"] = SC_HIGHWAY_SPEEDS_KMH.get(rt, SC_FALLBACK_SPEED_KMH)
                    data["travel_time"] = (data["length"] / 1000.0 / data["speed_kph"]) * 3600.0

        for _, _, _, data in G.edges(data=True, keys=True):
            data["travel_time"] = (data.get("travel_time", 0) + TURN_PENALTY_SECONDS) / 60.0

    return G


@st.cache_resource(show_spinner=False)
def get_osm_graph(center_lat, center_lon, dist_m, network_type):
    G = ox.graph_from_point(
        (center_lat, center_lon), dist=int(dist_m), network_type=network_type
    )
    return preprocess_network_speeds(G, network_type)


def build_coverage_matrix(
    candidates_subset, demand_subset, max_time, network_type="drive", use_network=False, G=None
):
    candidates_reset = candidates_subset.reset_index(drop=True).copy()
    demand_reset = demand_subset.reset_index(drop=True).copy()
    n_fac, n_dem = len(candidates_reset), len(demand_reset)

    if n_fac == 0 or n_dem == 0:
        return np.zeros((n_fac, n_dem), dtype=np.uint8), candidates_reset, demand_reset

    if use_network and G is not None:
        coverage = np.zeros((n_fac, n_dem), dtype=np.uint8)
        dem_nodes = snap_points_to_nodes(G, demand_reset["longitude"].to_numpy(), demand_reset["latitude"].to_numpy())
        fac_nodes = snap_points_to_nodes(G, candidates_reset["longitude"].to_numpy(), candidates_reset["latitude"].to_numpy())

        for i, origin in enumerate(fac_nodes):
            if origin is None:
                continue
            try:
                lengths = nx.single_source_dijkstra_path_length(
                    G, origin, cutoff=float(max_time), weight="travel_time"
                )
            except Exception:
                continue
            row = np.fromiter(
                ((lengths.get(n, np.inf) <= max_time) if n is not None else False for n in dem_nodes),
                dtype=np.bool_,
                count=n_dem,
            )
            coverage[i, :] = row.astype(np.uint8)

        return coverage, candidates_reset, demand_reset

    clat = candidates_reset["latitude"].to_numpy(dtype=float)[:, None]
    clon = candidates_reset["longitude"].to_numpy(dtype=float)[:, None]
    dlat = demand_reset["latitude"].to_numpy(dtype=float)[None, :]
    dlon = demand_reset["longitude"].to_numpy(dtype=float)[None, :]

    miles = (
        np.abs(dlat - clat) * 69.0
        + np.abs(dlon - clon) * 69.0 * np.cos(np.radians(clat))
    ) * CIRCUITY_FACTOR

    if network_type == "drive":
        tt = (miles / DEFAULT_DRIVING_SPEED) * 60.0
    else:
        tt = (miles * 1.60934 / WALKING_SPEED_KMH) * 60.0

    return (tt <= float(max_time)).astype(np.uint8), candidates_reset, demand_reset


def solve_maxcover(coverage_matrix, demand_weights, num_facilities):
    n_fac, n_dem = coverage_matrix.shape
    demand_weights = np.asarray(demand_weights, dtype=float)

    if n_fac == 0 or n_dem == 0 or int(num_facilities) <= 0:
        return [], 0.0, np.zeros(n_dem, dtype=bool)

    if int(num_facilities) >= n_fac:
        selected = list(range(n_fac))
        covered_mask = coverage_matrix.any(axis=0)
        covered_demand = float(demand_weights[covered_mask].sum())
        return selected, covered_demand, covered_mask

    model = LpProblem("Max_Coverage", LpMaximize)
    x = LpVariable.dicts("facility", range(n_fac), cat="Binary")
    y = LpVariable.dicts("covered", range(n_dem), cat="Binary")

    model += lpSum(demand_weights[j] * y[j] for j in range(n_dem))
    model += lpSum(x[i] for i in range(n_fac)) == int(num_facilities)

    for j in range(n_dem):
        coverers = np.where(coverage_matrix[:, j] == 1)[0]
        if coverers.size:
            model += y[j] <= lpSum(x[int(i)] for i in coverers)
        else:
            model += y[j] == 0

    model.solve(PULP_CBC_CMD(msg=0))
    selected = [i for i in range(n_fac) if x[i].varValue is not None and x[i].varValue > 0.5]

    if selected:
        covered_mask = coverage_matrix[selected, :].any(axis=0)
        covered_demand = float(demand_weights[covered_mask].sum())
    else:
        covered_mask = np.zeros(n_dem, dtype=bool)
        covered_demand = 0.0

    return selected, covered_demand, covered_mask


# ===========================
# MAIN APP
# ===========================
def main():
    st.title("🏥 South Carolina MHC Placement Decision Tool")
    st.markdown("**Optimizing healthcare accessibility for South Carolina's underserved communities.**")

    st.markdown(
        """<div class="instruction-box">
            <b>How to use:</b>
            Select a <b>County</b> to surface its ZIP codes first, or search any
            <b>ZIP Code</b> in South Carolina directly. Pick a <b>Target Variable</b>,
            adjust travel constraints, and click <b>Calculate Optimal Sites</b>.
        </div>""",
        unsafe_allow_html=True,
    )

    with st.expander("📖 Methodology Documentation"):
        st.markdown("""
            **Model Type:** Maximum Coverage Location Problem (MCLP)

            **Target Variable:** The optimization maximizes coverage of your selected
            demand variable. Each demand point is weighted by its value for that variable.

            **Network Analysis (optional):** OSM road network with posted speed limits.

            **Manhattan Distance (default):** Rectilinear distance x 1.2 circuity factor.
        """)

    st.divider()

    try:
        with st.spinner("Loading geospatial data..."):
            (
                zip_gdf,
                candidates_df,
                demand_df,
                county_gdf,
                zip_county_map,
                global_type_colors,
            ) = load_data(JSON_PATH)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    if county_gdf is None or len(county_gdf) == 0:
        st.error("County data not found in JSON. Please regenerate with county boundaries.")
        st.stop()

    available_targets = {
        label: col
        for label, col in TARGET_VARIABLE_OPTIONS.items()
        if col in demand_df.columns
    }

    with st.sidebar:
        st.header("Control Panel")

        run_analysis = st.button(
            "🚀 Calculate Optimal Sites",
            type="primary",
            use_container_width=True,
        )
        st.caption("Primary action")

        with st.expander("🎯 Target Variable", expanded=True):
            st.caption("Choose a category first, then the specific measure.")
            target_label, target_var, target_category = build_target_selector(available_targets)
            st.caption(f"Selected category: {target_category}")

        with st.expander("📍 Geographic Scope", expanded=True):
            county_names = sorted(county_gdf["county_name"].dropna().unique())
            county_name_to_fips = dict(zip(county_gdf["county_name"], county_gdf["COUNTY_FIPS"]))

            county_options = ["All South Carolina"] + county_names
            default_county_index = 1 if len(county_options) > 1 else 0

            selected_county_name = st.selectbox(
                "County (optional)",
                options=county_options,
                index=default_county_index,
                help="Select a county to filter ZIP codes to only those within that county.",
            )
            selected_county_fips = county_name_to_fips.get(selected_county_name)

            # Change 5: pass target_var so county ZIPs are ranked; county-only when selected
            zip_choices = get_ordered_zip_choices(
                zip_gdf,
                selected_county_fips,
                county_gdf,
                demand_df=demand_df,
                target_var=target_var,
            )

            zip_labels = zip_choices["zip_label"].tolist()

            if st.session_state.prev_county != selected_county_fips:
                default_zip = zip_choices.iloc[0]["ZIP_CODE"]
            else:
                default_zip = st.session_state.prev_zip
                if default_zip not in zip_choices["ZIP_CODE"].values:
                    default_zip = zip_choices.iloc[0]["ZIP_CODE"]

            default_zip_label = zip_choices.loc[
                zip_choices["ZIP_CODE"] == default_zip, "zip_label"
            ].iloc[0]

            selected_zip_label = st.selectbox(
                "ZIP Code",
                options=zip_labels,
                index=zip_labels.index(default_zip_label),
                help=(
                    "When a county is selected, only ZIP codes within that county are shown, "
                    "ranked by the selected target variable."
                ),
            )
            selected_zip = zip_choices.loc[
                zip_choices["zip_label"] == selected_zip_label, "ZIP_CODE"
            ].iloc[0]

        selected_zip_row = zip_gdf[zip_gdf["ZIP_CODE"] == selected_zip].iloc[0]
        selected_zip_display = build_zip_display(selected_zip_row)
        selected_zip_county_name = str(selected_zip_row.get("county_name", "")).strip()

        zip_geom = zip_gdf[zip_gdf["ZIP_CODE"] == selected_zip].iloc[0].geometry
        candidates_zip_all = get_candidates_in_zip(candidates_df, selected_zip, zip_geom)
        demand_in_zip = get_demand_in_zip(demand_df, selected_zip, zip_geom)

        with st.expander("⚙️ Model Constraints", expanded=True):
            available_site_types = sorted(candidates_zip_all["type"].dropna().unique())

            use_all_site_types = st.checkbox(
                "Use all eligible site types in this ZIP",
                value=True,
                help="Keeps the panel compact. Turn off only if you want to manually filter site types.",
            )

            if use_all_site_types:
                selected_types = available_site_types
                st.caption(f"{len(selected_types)} site types included")
            else:
                selected_types = st.multiselect(
                    "Choose site types",
                    options=available_site_types,
                    default=available_site_types,
                )
                st.caption(f"{len(selected_types)} of {len(available_site_types)} site types selected")

            use_network = st.toggle("Enable Road-Network Routing", value=DEFAULT_USE_NETWORK)

            c1, c2 = st.columns(2)
            with c1:
                travel_mode = st.radio("Travel Mode", options=["drive", "walk"], horizontal=True)
            with c2:
                default_time = 5 if travel_mode == "drive" else 10
                time_threshold = st.select_slider(
                    "Max Travel Time (min)",
                    options=[5, 10, 15, 20, 30, 45],
                    value=default_time,
                )

            candidates_in_zip = candidates_zip_all[
                candidates_zip_all["type"].isin(selected_types)
            ].copy()
            max_facilities = len(candidates_in_zip)

            num_facilities = st.number_input(
                "Target Number of Sites",
                min_value=1,
                max_value=max(1, max_facilities),
                value=min(DEFAULT_NUM_FACILITIES, max(1, max_facilities)),
                disabled=(max_facilities == 0),
            )

        with st.expander("🗺️ Display & advanced", expanded=False):
            map_theme = st.radio("Theme", options=["Light", "Dark"], horizontal=True)
            map_tiles = "CartoDB positron" if map_theme == "Light" else "CartoDB dark_matter"
            show_demand_preview = st.toggle(
                "Show block centroids before analysis",
                value=DEFAULT_SHOW_DEMAND_PREVIEW,
            )

    if st.session_state.prev_county != selected_county_fips:
        st.session_state.view_mode = "county" if selected_county_fips is not None else "zip"
        reset_analysis_state()
        st.session_state.prev_county = selected_county_fips
        st.session_state.prev_zip = selected_zip
    elif st.session_state.prev_zip != selected_zip:
        st.session_state.view_mode = "zip"
        reset_analysis_state()
        st.session_state.prev_zip = selected_zip

    if run_analysis:
        st.session_state.view_mode = "analysis"

    total_target = (
        float(demand_in_zip[target_var].sum())
        if len(demand_in_zip) and target_var in demand_in_zip.columns
        else 0.0
    )

    result_title = build_result_title(selected_zip_display, target_label, time_threshold, travel_mode)

    col_map, col_insights = st.columns([7, 3], gap="large")

    with col_insights:
        st.subheader("📊 Summary Statistics")
        if st.session_state.view_mode == "county" and selected_county_fips is not None:
            st.metric("County", selected_county_name)
        else:
            st.metric("ZIP Code", selected_zip_display)
        st.metric(f"Total {target_label}", f"{int(round(total_target)):,}")
        st.metric("Available Candidate Sites", f"{len(candidates_in_zip):,}")
        st.metric("Demand Points", f"{len(demand_in_zip):,}")

    if run_analysis:
        if max_facilities == 0:
            st.error("No candidate facilities available. Change ZIP or site types.")
        else:
            with st.spinner(f"Optimizing coverage for: {target_label}..."):
                G = None
                method_used = "Manhattan Distance"

                if use_network:
                    zip_center = zip_geom.centroid
                    net_type = "drive" if travel_mode == "drive" else "walk"
                    try:
                        graph_dist_m = estimate_required_graph_dist_m(
                            zip_center.y, zip_center.x, candidates_in_zip, demand_in_zip,
                            min_dist=15000, buffer_m=5000 if travel_mode == "drive" else 2000,
                        )
                        with st.spinner("Loading road network..."):
                            G = get_osm_graph(zip_center.y, zip_center.x, int(graph_dist_m), net_type)
                        method_used = "Road Network (OSM)"
                    except Exception as e:
                        st.warning(f"Road network failed, using Manhattan distance instead. {e}")

                coverage_matrix, candidates_reset, demand_reset = build_coverage_matrix(
                    candidates_in_zip, demand_in_zip, time_threshold,
                    network_type=travel_mode, use_network=use_network, G=G,
                )

                demand_weights = demand_reset[target_var].to_numpy(dtype=float)
                selected_indices, covered_pop, covered_mask = solve_maxcover(
                    coverage_matrix, demand_weights, int(num_facilities)
                )

                sel_fac = candidates_reset.iloc[selected_indices]
                site_metrics_lookup = compute_site_metrics(
                    selected_indices=selected_indices,
                    coverage_matrix=coverage_matrix,
                    demand_weights=demand_weights,
                    candidates_reset=candidates_reset,
                )

                st.session_state.update({
                    "analysis_complete": True,
                    "view_mode": "analysis",
                    "selected_facilities": sel_fac,
                    "coverage_matrix": coverage_matrix,
                    "demand_reset": demand_reset,
                    "candidates_reset": candidates_reset,
                    "covered_pop": covered_pop,
                    "covered_mask": covered_mask,
                    "method_used": method_used,
                    "selected_cand_ids": set(sel_fac["cand_idx"].astype(int).tolist()),
                    "covered_dem_ids": set(
                        demand_reset.loc[covered_mask, "dem_idx"].astype(int).tolist()
                    ),
                    "target_variable": target_var,
                    "target_label": target_label,
                    "site_metrics_lookup": site_metrics_lookup,
                    "analysis_title": result_title,
                })

    with col_map:
        if st.session_state.view_mode == "analysis" and st.session_state.analysis_complete:
            st.subheader(f"🗺️ {st.session_state.get('analysis_title', result_title)}")
            m = create_map(
                zip_gdf=zip_gdf,
                selected_zip=selected_zip,
                candidates_df=candidates_df,
                demand_df=demand_df,
                type_colors=global_type_colors,
                target_var=st.session_state.target_variable,
                target_label=st.session_state.target_label,
                selected_cand_ids=st.session_state.selected_cand_ids,
                covered_dem_ids=st.session_state.covered_dem_ids,
                site_metrics_lookup=st.session_state.site_metrics_lookup,
                show_demand_preview=True,
                selected_types=selected_types,
                tiles=map_tiles,
                county_gdf=county_gdf,
            )
            render_folium_map(m, key=f"map_{selected_zip}_done", height=640)

        elif st.session_state.view_mode == "zip":
            zip_title = selected_zip_display
            if selected_zip_county_name:
                zip_title = f"{selected_zip_display}, {selected_zip_county_name} County"
            st.subheader(f"🗺️ {zip_title}")
            m = create_map(
                zip_gdf=zip_gdf,
                selected_zip=selected_zip,
                candidates_df=candidates_df,
                demand_df=demand_df,
                type_colors=global_type_colors,
                target_var=target_var,
                target_label=target_label,
                show_demand_preview=show_demand_preview,
                selected_types=selected_types,
                tiles=map_tiles,
                county_gdf=county_gdf,
            )
            render_folium_map(m, key=f"map_{selected_zip}_pre", height=640)

        else:
            st.info(
                f"📍 **{selected_county_name} County**. Select a ZIP code from the sidebar or click a ZIP on the map."
            )
            m = create_county_overview_map(
                county_gdf,
                zip_gdf,
                zip_county_map,
                selected_county_fips,
                tiles=map_tiles,
                demand_df=demand_df,
                target_var=target_var,
                target_label=target_label,
            )
            map_data = st_folium(
                m,
                key=f"map_county_{selected_county_fips}",
                height=640,
                use_container_width=True,
                returned_objects=["last_active_drawing", "last_object_clicked_tooltip"],
            )

            clicked_tooltip = map_data.get("last_object_clicked_tooltip")
            if clicked_tooltip:
                try:
                    match = re.search(r"\b\d{5}\b", str(clicked_tooltip))
                    if match:
                        clicked_zip = match.group(0).zfill(5)
                        if clicked_zip in zip_choices["ZIP_CODE"].values:
                            st.session_state.prev_zip = clicked_zip
                            st.session_state.view_mode = "zip"
                            reset_analysis_state()
                            st.rerun()
                except Exception:
                    pass

    if st.session_state.analysis_complete and st.session_state.view_mode == "analysis":
        active_label = st.session_state.target_label

        with col_insights:
            cov_pop = float(st.session_state.covered_pop)
            pct = (cov_pop / total_target) * 100 if total_target > 0 else 0.0
            covered_count = (
                int(np.sum(st.session_state.covered_mask))
                if st.session_state.covered_mask is not None else 0
            )
            total_pts = (
                len(st.session_state.demand_reset)
                if st.session_state.demand_reset is not None else 0
            )
            st.metric(f"Covered {active_label}", f"{int(round(cov_pop)):,}")
            st.metric("Coverage Percentage", f"{pct:.1f}%")
            st.metric("Covered Demand Points", f"{covered_count} / {total_pts}")
            st.progress(min(max(pct / 100.0, 0.0), 1.0))
            st.caption(f"Method: {st.session_state.method_used}")

        st.divider()
        st.subheader(
            f"📍 Recommended Site Details — "
            f"{st.session_state.get('analysis_title', result_title)}"
        )

        sel_fac = st.session_state.selected_facilities
        cov_mat = st.session_state.coverage_matrix
        dem_reset = st.session_state.demand_reset
        cov_mask = st.session_state.covered_mask
        active_var = st.session_state.target_variable
        site_metrics_lookup = st.session_state.site_metrics_lookup

        st.caption(f"Covered demand points: {int(np.sum(cov_mask)):,} of {len(dem_reset):,}")

        demand_w = dem_reset[active_var].to_numpy(dtype=float)
        fac_cov = []
        for idx, row in sel_fac.iterrows():
            cand_idx = int(row["cand_idx"])
            if cand_idx in site_metrics_lookup:
                fac_cov.append(int(round(site_metrics_lookup[cand_idx]["covered_value"])))
            else:
                fac_cov.append(int(round(float(demand_w[cov_mat[idx, :] == 1].sum()))))

        df_display = sel_fac.copy()
        df_display[f"Covered {active_label}"] = fac_cov
        df_display = (
            df_display
            .sort_values(f"Covered {active_label}", ascending=False)
            .reset_index(drop=True)
        )
        df_display.insert(0, "Rank", range(1, len(df_display) + 1))

        show_cols = ["Rank"] + [
            c for c in ["name", "type", "address", f"Covered {active_label}"]
            if c in df_display.columns
        ]
        st.dataframe(df_display[show_cols], use_container_width=True, hide_index=True)

        st.subheader("📥 Export Results")
        c1, c2 = st.columns(2)

        with c1:
            exp_cols = [
                c for c in ["facility_id", "name", "type", "address", "latitude", "longitude"]
                if c in sel_fac.columns
            ]
            st.download_button(
                "Download Proposed Sites (CSV)",
                sel_fac[exp_cols].to_csv(index=False),
                f"proposed_sites_{selected_zip}.csv",
                "text/csv",
                key="csv_dl",
            )

        with c2:
            gdf_sel = gpd.GeoDataFrame(
                sel_fac,
                geometry=gpd.points_from_xy(sel_fac["longitude"], sel_fac["latitude"]),
                crs="EPSG:4326",
            )
            st.download_button(
                "Download Proposed Sites (GeoJSON)",
                gdf_sel.to_json(),
                f"proposed_sites_{selected_zip}.geojson",
                "application/geo+json",
                key="geojson_dl",
            )


if __name__ == "__main__":
    main()

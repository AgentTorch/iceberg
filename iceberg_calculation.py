import os
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List, Any, Optional
from tqdm import tqdm
import geopandas as gpd

# --- Utility Functions ---
def clean_soc_code(soc_code: str) -> str:
    """Clean SOC code by removing hyphens."""
    return str(soc_code).replace("-", "") if pd.notna(soc_code) else ""

def convert_datatypes(df: pd.DataFrame, numeric_cols: List[str] = ["TOT_EMP", "A_MEAN", "A_MEDIAN"]) -> pd.DataFrame:
    """Convert string columns to numeric, handling asterisks and commas."""
    df = df.copy()
    for col in numeric_cols:
        if col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].replace("*", np.nan)
                df[col] = pd.to_numeric(df[col].str.replace(",", ""), errors="coerce")
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def categorize_automation_risk(score: float) -> str:
    """Categorize automation risk based on score."""
    if pd.isna(score):
        return "Unknown"
    if score >= 80:
        return "Very High"
    elif score >= 60:
        return "High"
    elif score >= 40:
        return "Moderate"
    elif score >= 20:
        return "Low"
    else:
        return "Very Low"

def load_soc_mappings(soc_mapping_file: str) -> Dict[str, Dict[str, str]]:
    """
    Load SOC code mappings from JSON file.
    
    Args:
        soc_mapping_file: Path to soc_mapping.json
        
    Returns:
        Dictionary with minor and major group mappings
    """
    print(f"Loading SOC mappings from {soc_mapping_file}...")
    try:
        with open(soc_mapping_file, "r") as f:
            soc_data = json.load(f)
        
        minor_mappings = {}
        major_mappings = {}
        for detailed_code, info in soc_data.items():
            minor_code = info["minor_code"].replace("-", "")[:3]
            minor_title = info["minor_title"]
            major_code = info["major_code"].replace("-", "")[:2]
            major_title = info["major_title"]
            if minor_code and minor_title and minor_code not in minor_mappings:
                minor_mappings[minor_code] = minor_title
            if major_code and major_title and major_code not in major_mappings:
                major_mappings[major_code] = major_title
        
        print(f"Loaded {len(minor_mappings)} minor and {len(major_mappings)} major group mappings")
        return {"minor": minor_mappings, "major": major_mappings}
    except Exception as e:
        print(f"Error loading SOC mappings: {e}. Using empty mappings.")
        return {"minor": {}, "major": {}}

# --- Capture Phase ---
def capture_employment_data(employment_file: str, state_name: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[float]]:
    """Load and filter employment data for a state or nationally, preventing double counting."""
    print(f"Loading employment data{' for ' + state_name if state_name else ''}...")
    emp_df = pd.read_csv(employment_file)
    
    if state_name:
        df = emp_df[emp_df["AREA_TITLE"] == state_name].copy()
        if df.empty:
            raise ValueError(f"No data found for {state_name}")
        print(f"Found {len(df)} occupation records for {state_name}")
    else:
        df = emp_df[emp_df["AREA_TITLE"] != "U.S."].copy()  # National data excludes aggregate U.S. row
        print(f"Found {len(df)} occupation records for national data")

    # Extract official total employment
    official_total = None
    if "total" in df["O_GROUP"].values:
        total_row = df[df["O_GROUP"] == "total"]
        if not total_row.empty:
            official_total = pd.to_numeric(total_row["TOT_EMP"].iloc[0], errors="coerce")
            print(f"Official total employment: {official_total}")

    # Filter to detailed occupations, fallback to broad/major if needed
    if "detailed" in df["O_GROUP"].values:
        df_filtered = df[df["O_GROUP"] == "detailed"].copy()
        if len(df_filtered) < len(df) * 0.7 and "broad" in df["O_GROUP"].values:
            print("Not enough detailed data, adding broad occupations...")
            major_codes = set(df[df["O_GROUP"] == "major"]["OCC_CODE"].astype(str))
            detailed_prefixes = set([code[:2] for code in df_filtered["OCC_CODE"].astype(str)])
            missing_detailed = [code for code in major_codes if code[:2] not in detailed_prefixes]
            if missing_detailed:
                broad_to_add = df[
                    (df["O_GROUP"] == "broad") & 
                    (df["OCC_CODE"].astype(str).str[:2].isin([code[:2] for code in missing_detailed]))
                ]
                df_filtered = pd.concat([df_filtered, broad_to_add])
    elif "broad" in df["O_GROUP"].values:
        print("Using broad occupations...")
        df_filtered = df[df["O_GROUP"] == "broad"].copy()
    else:
        print("Using major occupations or all available...")
        df_filtered = df[df["O_GROUP"] != "total"].copy() if "total" in df["O_GROUP"].values else df.copy()

    # Clean and convert data
    df_filtered["SOC_Code_Cleaned"] = df_filtered["OCC_CODE"].apply(clean_soc_code)
    df_filtered = convert_datatypes(df_filtered)

    # Validate total employment
    calculated_total = df_filtered["TOT_EMP"].sum()
    print(f"Calculated total employment: {calculated_total}")
    if official_total:
        discrepancy_pct = abs(calculated_total - official_total) / official_total * 100
        if discrepancy_pct > 5:
            print(f"WARNING: Calculated total differs by {discrepancy_pct:.1f}% from official total")
        else:
            print(f"Validation successful: Calculated total within {discrepancy_pct:.1f}%")

    return df_filtered, official_total

def capture_tech_intensity_data(tech_intensity_file: str) -> pd.DataFrame:
    """Load and process tech intensity data, calculating automation susceptibility."""
    print("Loading tech intensity data...")
    tech_df = pd.read_csv(tech_intensity_file)

    # Normalize tech intensity and calculate susceptibility
    max_tech_intensity = tech_df["tech_intensity_score"].max()
    tech_df["tech_intensity_normalized"] = tech_df["tech_intensity_score"] / max_tech_intensity
    tech_df["automation_susceptibility"] = (
        (tech_df["tech_intensity_normalized"] * 0.8) + (tech_df["hot_tech_ratio"] * 0.2)
    ) * 100
    tech_df["automation_susceptibility"] = tech_df["automation_susceptibility"].clip(0, 100)
    tech_df["automation_risk_category"] = tech_df["automation_susceptibility"].apply(categorize_automation_risk)

    # Clean SOC codes
    soc_col = "SOC_Detailed" if "SOC_Detailed" in tech_df.columns else "SOC_Code"
    tech_df["SOC_Code_Cleaned"] = tech_df[soc_col].apply(clean_soc_code)

    return tech_df

def capture_mcp_data(zapier_apps_file: str) -> pd.DataFrame:
    """Load MCP server data from Zapier file."""
    print(f"Loading MCP server data from {zapier_apps_file}...")
    try:
        if zapier_apps_file.endswith(".json"):
            with open(zapier_apps_file, "r") as f:
                data = json.load(f)
            zapier_df = pd.DataFrame(data.get("zapier_apps_by_detailed_soc", data.get("zapier_apps_by_soc", [])))
        else:
            zapier_df = pd.read_csv(zapier_apps_file)
    except Exception as e:
        print(f"Error loading MCP data: {e}. Using default empty DataFrame.")
        zapier_df = pd.DataFrame(columns=["major_group_code", "estimated_apps"])
    return zapier_df

# --- Analyze Phase ---
def analyze_workforce_data(emp_df: pd.DataFrame, tech_df: pd.DataFrame, national_emp_df: pd.DataFrame) -> pd.DataFrame:
    """Merge employment and tech intensity data, filling missing values."""
    print("Merging employment and tech intensity data...")
    print("Emp Df shape: ", emp_df.shape, "tech df shape: ", tech_df.shape)
    original_total = emp_df["TOT_EMP"].sum()
    
    # Remove duplicates before merging
    emp_df = emp_df.drop_duplicates(subset=["SOC_Code_Cleaned"])
    tech_df = tech_df.drop_duplicates(subset=["SOC_Code_Cleaned"])
    
    merged_df = pd.merge(
        emp_df,
        tech_df[["SOC_Code_Cleaned", "tech_intensity_score", "automation_susceptibility", "automation_risk_category"]],
        on="SOC_Code_Cleaned",
        how="left"
    )

    # Check for duplication
    merged_total = merged_df["TOT_EMP"].sum()
    if merged_total > original_total * 1.1:
        print(f"WARNING: Duplicates detected. Removing {len(merged_df) - len(emp_df)} rows.")
        merged_df = merged_df.drop_duplicates(subset=["SOC_Code_Cleaned"])
        print(f"Employment total after fixing: {merged_df['TOT_EMP'].sum()}")

    # Fill missing automation scores
    if merged_df["automation_susceptibility"].isna().any():
        missing_count = merged_df["automation_susceptibility"].isna().sum()
        print(f"Filling {missing_count} missing automation scores with median.")
        median_score = tech_df["automation_susceptibility"].median()
        merged_df["automation_susceptibility"].fillna(median_score, inplace=True)
        merged_df["automation_risk_category"] = merged_df["automation_susceptibility"].apply(categorize_automation_risk)

    # Fill missing wages using national data
    missing_wages = merged_df["A_MEAN"].isna() | (merged_df["A_MEAN"] == 0)
    if missing_wages.any():
        print(f"Handling {missing_wages.sum()} missing/zero wages...")
        national_data = national_emp_df[national_emp_df["AREA_TITLE"] == "U.S."].copy()
        if not national_data.empty:
            national_data["SOC_Code_Cleaned"] = national_data["OCC_CODE"].apply(clean_soc_code)
            national_data = convert_datatypes(national_data)
            wage_map = national_data.set_index("SOC_Code_Cleaned")["A_MEAN"].to_dict()
            merged_df.loc[missing_wages, "A_MEAN"] = merged_df.loc[missing_wages, "SOC_Code_Cleaned"].map(wage_map)
        
        # Fill remaining with state median
        still_missing = merged_df["A_MEAN"].isna() | (merged_df["A_MEAN"] == 0)
        median_wage = merged_df[~still_missing]["A_MEAN"].median() or 50000
        merged_df.loc[still_missing, "A_MEAN"] = median_wage
        print(f"Used median wage ${median_wage:.2f} for remaining missing values.")

    return merged_df

def analyze_mcp_coverage(merged_df: pd.DataFrame, zapier_df: pd.DataFrame) -> pd.DataFrame:
    """Integrate MCP server data and calculate enhanced automation risk."""
    print("Analyzing MCP server coverage...")
    merged_df["major_group"] = merged_df["SOC_Code_Cleaned"].str[:2]

    # Map Zapier apps to major groups
    major_group_apps = {}
    soc_col = "major_group_code" if "major_group_code" in zapier_df.columns else "soc_code"
    if soc_col in zapier_df.columns:
        for _, row in zapier_df.iterrows():
            major_code = clean_soc_code(row[soc_col])[:2]
            major_group_apps[major_code] = row["estimated_apps"]
        merged_df["estimated_zapier_apps_major"] = merged_df["major_group"].map(major_group_apps)
    else:
        print("WARNING: No valid MCP data. Setting apps to 0.")
        merged_df["estimated_zapier_apps_major"] = 0

    # Distribute apps by employment proportion
    merged_df["estimated_zapier_apps_major"] = merged_df["estimated_zapier_apps_major"].fillna(0)
    major_group_emp = merged_df.groupby("major_group")["TOT_EMP"].sum().to_dict()
    merged_df["major_group_emp_total"] = merged_df["major_group"].map(major_group_emp)
    merged_df["emp_proportion"] = merged_df["TOT_EMP"] / merged_df["major_group_emp_total"].replace(0, np.nan)
    merged_df["estimated_zapier_apps"] = merged_df["estimated_zapier_apps_major"] * merged_df["emp_proportion"]
    merged_df["estimated_zapier_apps"] = merged_df["estimated_zapier_apps"].fillna(0)

    # Calculate apps per worker
    merged_df["zapier_apps_per_worker"] = merged_df["estimated_zapier_apps"] / merged_df["TOT_EMP"].replace(0, np.nan)
    merged_df["zapier_apps_per_worker"] = merged_df["zapier_apps_per_worker"].fillna(0)

    # Normalize apps and calculate enhanced risk
    max_apps = merged_df["estimated_zapier_apps"].max()
    merged_df["zapier_apps_normalized"] = merged_df["estimated_zapier_apps"] / max_apps if max_apps > 0 else 0
    merged_df["automation_susceptibility_norm"] = merged_df["automation_susceptibility"] / 100

    zapier_apps_scaled = merged_df["zapier_apps_per_worker"].clip(upper=0.01) / 0.01
    
    weights = {"automation_susceptibility": 0.75, "zapier_coverage": 0.25}
    merged_df["enhanced_automation_risk"] = (
        (merged_df["automation_susceptibility_norm"] * weights["automation_susceptibility"]) +
        (merged_df["zapier_apps_normalized"] * weights["zapier_coverage"])
    ) * 100
    merged_df["enhanced_automation_risk"] = merged_df["enhanced_automation_risk"].clip(0, 100)
    merged_df["enhanced_risk_category"] = merged_df["enhanced_automation_risk"].apply(categorize_automation_risk)

    return merged_df

# --- Measure Phase ---
def measure_iceberg_indices(merged_df: pd.DataFrame, soc_mappings: Dict[str, Dict[str, str]], potential_threshold: int = 50) -> Tuple[pd.DataFrame, float]:
    """Calculate Iceberg Indices for minor groups and state-wide."""
    print("Calculating Iceberg Indices...")
    
    # Debug: Check merged_df
    print(f"merged_df size: {len(merged_df)}")
    print(f"merged_df columns: {merged_df.columns.tolist()}")
    if merged_df.empty:
        print("ERROR: merged_df is empty. Returning empty industry_risk.")
        return pd.DataFrame(columns=["minor_group", "minor_group_name", "TOT_EMP", "economic_value",
                                    "automation_susceptibility", "enhanced_automation_risk",
                                    "is_potentially_at_risk", "is_currently_at_risk",
                                    "estimated_zapier_apps", "zapier_apps_per_worker",
                                    "at_risk_soc_codes", "potential_econ_value_at_risk",
                                    "current_econ_value_at_risk", "minor_potential_index",
                                    "minor_iceberg_index", "automation_gap", "weighted_iceberg_index"]), 0.0
    
    # Calculate economic value
    merged_df["economic_value"] = merged_df["TOT_EMP"] * merged_df["A_MEAN"]
    total_econ_value = merged_df["economic_value"].sum()
    if total_econ_value == 0:
        print("WARNING: Zero economic value. Using employment as proxy.")
        merged_df["economic_value"] = merged_df["TOT_EMP"]
        total_econ_value = merged_df["economic_value"].sum()

    # Flag risks
    merged_df["is_potentially_at_risk"] = merged_df["automation_susceptibility"] > potential_threshold
    merged_df["is_currently_at_risk"] = merged_df["enhanced_automation_risk"] > 60
    
    # Calculate potential_index per SOC code
    merged_df["potential_index"] = merged_df["is_potentially_at_risk"].apply(lambda x: 100.0 if x else 0.0)
    merged_df["econ_potential_index"] = (merged_df["automation_susceptibility"] * (merged_df["economic_value"])/total_econ_value)

    merged_df["econ_potential_index"] = merged_df["econ_potential_index"] / merged_df["econ_potential_index"].max()

    # Create minor_group and minor_group_name
    if "SOC_Code_Cleaned" not in merged_df.columns:
        print("ERROR: SOC_Code_Cleaned missing from merged_df. Returning empty industry_risk.")
        return pd.DataFrame(columns=["minor_group", "minor_group_name", "TOT_EMP", "economic_value",
                                    "automation_susceptibility", "enhanced_automation_risk",
                                    "is_potentially_at_risk", "is_currently_at_risk",
                                    "estimated_zapier_apps", "zapier_apps_per_worker",
                                    "at_risk_soc_codes", "potential_econ_value_at_risk",
                                    "current_econ_value_at_risk", "minor_potential_index",
                                    "minor_iceberg_index", "automation_gap", "weighted_iceberg_index"]), total_econ_value

    # Validate SOC_Code_Cleaned
    print(f"Missing SOC_Code_Cleaned: {merged_df['SOC_Code_Cleaned'].isna().sum()}")
    print(f"Sample SOC_Code_Cleaned: {merged_df['SOC_Code_Cleaned'].head().tolist()}")
    invalid_soc = merged_df["SOC_Code_Cleaned"].isna() | merged_df["SOC_Code_Cleaned"].str.strip().eq("")
    if invalid_soc.any():
        print(f"WARNING: Found {invalid_soc.sum()} invalid SOC_Code_Cleaned values. Dropping these rows.")
        merged_df = merged_df[~invalid_soc].copy()

    merged_df["minor_group"] = merged_df["SOC_Code_Cleaned"].str[:3]
    merged_df["minor_group_name"] = merged_df["minor_group"].map(soc_mappings["minor"]).fillna(merged_df["minor_group"])
    
    # Debug: Verify minor_group
    print(f"Missing minor_group: {merged_df['minor_group'].isna().sum()}")
    print(f"Unique minor groups: {len(merged_df['minor_group'].unique())}")
    print(f"Sample minor groups: {merged_df['minor_group'].head().tolist()}")
    print(f"merged_df columns after adding minor_group: {merged_df.columns.tolist()}")

    # Check if minor_group is valid
    if merged_df["minor_group"].isna().all() or merged_df["minor_group"].str.strip().eq("").all():
        print("ERROR: No valid minor_group values in merged_df. Returning empty industry_risk.")
        return pd.DataFrame(columns=["minor_group", "minor_group_name", "TOT_EMP", "economic_value",
                                    "automation_susceptibility", "enhanced_automation_risk",
                                    "is_potentially_at_risk", "is_currently_at_risk",
                                    "estimated_zapier_apps", "zapier_apps_per_worker",
                                    "at_risk_soc_codes", "potential_econ_value_at_risk",
                                    "current_econ_value_at_risk", "minor_potential_index",
                                    "minor_iceberg_index", "automation_gap", "weighted_iceberg_index"]), total_econ_value

    # Debug: Check for missing values
    print(f"Missing automation_susceptibility: {merged_df['automation_susceptibility'].isna().sum()}")
    print(f"Missing is_potentially_at_risk: {merged_df['is_potentially_at_risk'].isna().sum()}")

    def get_at_risk_soc_codes(group):
        """Safely get comma-separated SOC codes where is_potentially_at_risk is True."""
        try:
            at_risk = group[group["is_potentially_at_risk"]]["SOC_Code_Cleaned"]
            return ",".join(at_risk.dropna().astype(str)) if not at_risk.empty else ""
        except Exception as e:
            print(f"Error in get_at_risk_soc_codes: {e}")
            return ""

    metrics = {
        "TOT_EMP": "sum",
        "economic_value": "sum",
        "automation_susceptibility": "mean",
        "enhanced_automation_risk": "mean",
        "is_potentially_at_risk": "sum",
        "is_currently_at_risk": "sum",
        "estimated_zapier_apps": "sum",
        "zapier_apps_per_worker": "mean",
        "at_risk_soc_codes": get_at_risk_soc_codes
    }
    
    try:
        industry_risk = merged_df.groupby(["minor_group", "minor_group_name"]).apply(
            lambda g: pd.Series({
                "TOT_EMP": g["TOT_EMP"].sum(),
                "economic_value": g["economic_value"].sum(),
                "automation_susceptibility": g["automation_susceptibility"].mean(),
                "enhanced_automation_risk": g["enhanced_automation_risk"].mean(),
                "is_potentially_at_risk": g["is_potentially_at_risk"].sum(),
                "is_currently_at_risk": g["is_currently_at_risk"].sum(),
                "estimated_zapier_apps": g["estimated_zapier_apps"].sum(),
                "zapier_apps_per_worker": g["zapier_apps_per_worker"].mean(),
                "at_risk_soc_codes": get_at_risk_soc_codes(g)
            })
        ).reset_index()
    except Exception as e:
        print(f"Error during industry_risk aggregation: {e}")
        industry_risk = pd.DataFrame(columns=["minor_group", "minor_group_name", "TOT_EMP", "economic_value",
                                             "automation_susceptibility", "enhanced_automation_risk",
                                             "is_potentially_at_risk", "is_currently_at_risk",
                                             "estimated_zapier_apps", "zapier_apps_per_worker",
                                             "at_risk_soc_codes"])

    # Debug: Print number of minor groups
    print(f"Number of minor groups in industry_risk: {len(industry_risk)}")
    if not industry_risk.empty:
        print(f"Columns in industry_risk: {industry_risk.columns.tolist()}")

    # Calculate economic value at risk
    for minor_group in industry_risk["minor_group"]:
        if pd.isna(minor_group):
            continue
        group_data = merged_df[merged_df["minor_group"] == minor_group]
        industry_risk.loc[industry_risk["minor_group"] == minor_group, "potential_econ_value_at_risk"] = \
            group_data[group_data["is_potentially_at_risk"]]["economic_value"].sum()
        industry_risk.loc[industry_risk["minor_group"] == minor_group, "current_econ_value_at_risk"] = \
            group_data[group_data["is_currently_at_risk"]]["economic_value"].sum()

    # Compute indices
    industry_risk["minor_potential_index"] = (industry_risk["potential_econ_value_at_risk"] / industry_risk["economic_value"].replace(0, np.nan)) * 100
    industry_risk["minor_iceberg_index"] = (industry_risk["current_econ_value_at_risk"] / industry_risk["economic_value"].replace(0, np.nan)) * 100
    industry_risk["automation_gap"] = industry_risk["minor_potential_index"] - industry_risk["minor_iceberg_index"]

    # State-wide Iceberg Index
    industry_risk["weighted_iceberg_index"] = industry_risk["minor_iceberg_index"] * (industry_risk["economic_value"] / total_econ_value)
    state_iceberg_index = industry_risk["weighted_iceberg_index"].sum()
    print(f"State-wide Iceberg Index: {state_iceberg_index:.2f}%")

    return industry_risk, total_econ_value

def measure_summary(state_name: str, merged_df: pd.DataFrame, industry_risk: pd.DataFrame, 
                    total_econ_value: float, potential_at_risk: float, current_at_risk: float, 
                    risk_threshold: int) -> Dict[str, Any]:
    """Create summary of automation risk metrics."""
    potential_ratio = (potential_at_risk / total_econ_value) * 100 if total_econ_value > 0 else 0
    iceberg_ratio = (current_at_risk / total_econ_value) * 100 if total_econ_value > 0 else 0
    industry_risk_sorted = industry_risk.sort_values("minor_iceberg_index", ascending=False)
    highest_risk = industry_risk_sorted["minor_group_name"].head(3).dropna().tolist()
    lowest_risk = industry_risk_sorted["minor_group_name"].tail(3).dropna().tolist()

    summary = {
        "state": state_name,
        "potential_ratio": potential_ratio,
        "iceberg_ratio": iceberg_ratio,
        "automation_gap": potential_ratio - iceberg_ratio,
        "risk_threshold_used": risk_threshold,
        "total_occupations": len(merged_df),
        "occupations_at_potential_risk": merged_df["is_potentially_at_risk"].sum(),
        "occupations_at_current_risk": merged_df["is_currently_at_risk"].sum(),
        "total_employment": merged_df["TOT_EMP"].sum(),
        "potential_employment_at_risk": merged_df.loc[merged_df["is_potentially_at_risk"], "TOT_EMP"].sum(),
        "current_employment_at_risk": merged_df.loc[merged_df["is_currently_at_risk"], "TOT_EMP"].sum(),
        "pct_potential_employment_at_risk": (merged_df.loc[merged_df["is_potentially_at_risk"], "TOT_EMP"].sum() / merged_df["TOT_EMP"].sum()) * 100,
        "pct_current_employment_at_risk": (merged_df.loc[merged_df["is_currently_at_risk"], "TOT_EMP"].sum() / merged_df["TOT_EMP"].sum()) * 100,
        "economic_value_total": total_econ_value,
        "potential_economic_value_at_risk": potential_at_risk,
        "current_economic_value_at_risk": current_at_risk,
        "pct_potential_economic_value_at_risk": potential_ratio,
        "pct_current_economic_value_at_risk": iceberg_ratio,
        "highest_risk_industries": highest_risk,
        "lowest_risk_industries": lowest_risk
    }

    print("\n--- SUMMARY STATISTICS ---")
    for key, value in summary.items():
        if isinstance(value, (int, float)):
            print(f"{key.replace('_', ' ').title()}: {value:.2f}{'%' if 'pct' in key or 'ratio' in key else ''}")
        elif isinstance(value, list):
            print(f"{key.replace('_', ' ').title()}: {', '.join(value)}")
        else:
            print(f"{key.replace('_', ' ').title()}: {value}")

    return summary

# --- Visualization Functions ---
def visualize_industry_risk(state_name: str, industry_risk: pd.DataFrame, output_dir: str, top_n: int = 15) -> None:
    """Create industry-level visualizations for minor groups."""
    os.makedirs(output_dir, exist_ok=True)
    try:
        # Limit to top N minor groups to avoid overcrowding
        industry_risk_top = industry_risk.sort_values("potential_econ_value_at_risk", ascending=False).head(top_n)
        
        plt.figure(figsize=(14, 10))
        melted = pd.melt(
            industry_risk_top[["minor_group_name", "minor_potential_index", "minor_iceberg_index"]],
            id_vars=["minor_group_name"],
            value_vars=["minor_potential_index", "minor_iceberg_index"],
            var_name="Metric",
            value_name="Score"
        )
        melted["Metric"] = melted["Metric"].replace({
            "minor_potential_index": "Potential Risk",
            "minor_iceberg_index": "Current Risk (Iceberg Index)"
        })
        sns.barplot(data=melted, x="Score", y="minor_group_name", hue="Metric", palette=["#ff9999", "#0066cc"])
        plt.title(f"{state_name} Minor Group Automation Risk: Potential vs Current (Top {top_n})", fontsize=14)
        plt.xlabel("Percentage of Economic Value at Risk (%)", fontsize=12)
        plt.ylabel("Minor Occupational Group", fontsize=12)
        plt.legend(title="", loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{state_name.lower().replace(' ', '_')}_minor_group_risk_comparison.png"), dpi=300)
        plt.close()
        
        if "automation_gap" in industry_risk_top.columns:
            plt.figure(figsize=(14, 10))
            gap_data = industry_risk_top[["minor_group_name", "automation_gap"]].sort_values("automation_gap", ascending=False)
            colors = ["#0066cc" if x > 0 else "#ff9999" for x in gap_data["automation_gap"]]
            sns.barplot(data=gap_data, x="automation_gap", y="minor_group_name", palette=colors)
            plt.title(f"{state_name} Minor Group Automation Gap: Potential - Current (Top {top_n})", fontsize=14)
            plt.xlabel("Automation Gap (Percentage Points)", fontsize=12)
            plt.ylabel("Minor Occupational Group", fontsize=12)
            plt.axvline(x=0, color="black", linestyle="--")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{state_name.lower().replace(' ', '_')}_minor_group_automation_gap.png"), dpi=300)
            plt.close()
    except Exception as e:
        print(f"Error creating minor group visualizations: {e}")

def visualize_national_results(national_summary_df: pd.DataFrame, national_industry_df: pd.DataFrame, output_dir: str, top_n: int = 15) -> None:
    """Create national-level visualizations."""
    os.makedirs(output_dir, exist_ok=True)

    # Choropleth maps
    try:
        us_states = gpd.read_file("https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json")
        us_states["state"] = us_states["name"]
        map_data = us_states.merge(national_summary_df, on="state", how="left")
        
        for metric, title in [
            ("iceberg_ratio", "Current Automation Risk by State (%)"),
            ("potential_ratio", "Potential Automation Risk by State (%)"),
            ("automation_gap", "Automation Gap by State (Potential - Current)")
        ]:
            if metric not in map_data.columns and metric == "automation_gap":
                map_data["automation_gap"] = map_data["potential_ratio"] - map_data["iceberg_ratio"]
            if metric in map_data.columns:
                fig, ax = plt.subplots(1, 1, figsize=(15, 10))
                map_data.plot(column=metric, ax=ax, legend=True, cmap="YlOrRd" if metric != "automation_gap" else "PuOr",
                              missing_kwds={"color": "lightgrey"})
                plt.title(f"Project Iceberg: {title}", fontsize=16)
                plt.savefig(os.path.join(output_dir, f"national_{metric}_map.png"), dpi=300, bbox_inches="tight")
                plt.close()
    except Exception as e:
        print(f"Error creating map visualizations: {e}")

    # Minor group comparison
    try:
        industry_avg = national_industry_df.groupby(["minor_group", "minor_group_name"]).agg({
            "TOT_EMP": "sum",
            "potential_econ_value_at_risk": "sum",
            "minor_potential_index": "mean",
            "minor_iceberg_index": "mean",
            "zapier_apps_per_worker": "mean",
            "automation_gap": "mean"
        }).reset_index().sort_values("potential_econ_value_at_risk", ascending=False)
        
        # Limit to top N for visualization
        industry_avg_top = industry_avg.head(top_n)
        
        plt.figure(figsize=(14, 10))
        melted = pd.melt(
            industry_avg_top[["minor_group_name", "minor_potential_index", "minor_iceberg_index"]],
            id_vars=["minor_group_name"],
            value_vars=["minor_potential_index", "minor_iceberg_index"],
            var_name="Metric",
            value_name="Score"
        )
        melted["Metric"] = melted["Metric"].replace({
            "minor_potential_index": "Potential Risk",
            "minor_iceberg_index": "Current Risk (Iceberg Index)"
        })
        sns.barplot(data=melted, x="Score", y="minor_group_name", hue="Metric", palette=["#ff9999", "#0066cc"])
        plt.title(f"Average Minor Group Automation Risk: Potential vs Current (Top {top_n})", fontsize=16)
        plt.xlabel("Percentage of Economic Value at Risk (%)", fontsize=14)
        plt.ylabel("Minor Occupational Group", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "minor_group_comparison_enhanced.png"), dpi=300)
        plt.close()
        
        if "automation_gap" in industry_avg_top.columns:
            plt.figure(figsize=(14, 10))
            gap_data = industry_avg_top[["minor_group_name", "automation_gap"]].sort_values("automation_gap", ascending=False)
            colors = ["#0066cc" if x > 0 else "#ff9999" for x in gap_data["automation_gap"]]
            sns.barplot(data=gap_data, x="automation_gap", y="minor_group_name", palette=colors)
            plt.title(f"Minor Group Automation Gap: Potential - Current (Top {top_n})", fontsize=16)
            plt.xlabel("Automation Gap (Percentage Points)", fontsize=14)
            plt.ylabel("Minor Occupational Group", fontsize=14)
            plt.axvline(x=0, color="black", linestyle="--")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "minor_group_automation_gap.png"), dpi=300)
            plt.close()
    except Exception as e:
        print(f"Error creating minor group visualizations: {e}")

    # MCP coverage correlation
    try:
        mcp_by_group = national_industry_df.groupby(["minor_group", "minor_group_name"]).agg({
            "minor_iceberg_index": "mean",
            "zapier_apps_per_worker": "mean",
            "TOT_EMP": "sum"
        }).reset_index()
        plt.figure(figsize=(14, 10))
        sizes = mcp_by_group["TOT_EMP"] / mcp_by_group["TOT_EMP"].max() * 1000
        plt.scatter(mcp_by_group["zapier_apps_per_worker"], mcp_by_group["minor_iceberg_index"],
                    s=sizes, alpha=0.7, c=mcp_by_group["minor_iceberg_index"], cmap="YlOrRd")
        for i, row in mcp_by_group.iterrows():
            plt.annotate(row["minor_group_name"], (row["zapier_apps_per_worker"], row["minor_iceberg_index"]), fontsize=8)
        plt.colorbar(label="Current Automation Risk")
        plt.title("Relationship Between MCP Server Coverage and Automation Risk (Minor Groups)", fontsize=16)
        plt.xlabel("MCP Servers per Worker (Zapier Apps)", fontsize=14)
        plt.ylabel("Minor Group Iceberg Index (%)", fontsize=14)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "mcp_coverage_correlation_minor.png"), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error creating MCP coverage visualization: {e}")

# --- Main Workflow ---
def calculate_state_iceberg_index(
    state_name: str, employment_file: str, tech_intensity_file: str, zapier_apps_file: str,
    soc_mapping_file: str, potential_threshold: int = 50, output_dir: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    print(f"\n=== CALCULATING AUTOMATION RISK FOR {state_name.upper()} ===")
    
    soc_mappings = load_soc_mappings(soc_mapping_file)
    emp_df, official_total = capture_employment_data(employment_file, state_name)
    tech_df = capture_tech_intensity_data(tech_intensity_file)
    zapier_df = capture_mcp_data(zapier_apps_file)
    national_emp_df = pd.read_csv(employment_file)

    merged_df = analyze_workforce_data(emp_df, tech_df, national_emp_df)
    merged_df = analyze_mcp_coverage(merged_df, zapier_df)
    
    # Check for duplicate SOC codes
    duplicates = merged_df[merged_df.duplicated(subset=["SOC_Code_Cleaned"], keep=False)]
    if not duplicates.empty:
        print(f"WARNING: Found {len(duplicates)} duplicate SOC codes in merged_df:")
        print(duplicates[["SOC_Code_Cleaned", "OCC_TITLE", "TOT_EMP"]])
        merged_df = merged_df.drop_duplicates(subset=["SOC_Code_Cleaned"])
        print(f"Removed duplicates. New merged_df size: {len(merged_df)}")
    
    # Debug: Print data summary for Alabama
    if state_name.lower() == "alabama":
        print(f"\nDebugging Alabama:")
        print(f"Number of occupations in merged_df: {len(merged_df)}")
        print(f"merged_df columns: {merged_df.columns.tolist()}")
        print(f"Sample SOC codes: {merged_df['SOC_Code_Cleaned'].head().tolist()}")
        print(f"Sample merged_df rows:\n{merged_df[['SOC_Code_Cleaned', 'OCC_TITLE', 'TOT_EMP']].head()}")

    industry_risk, total_econ_value = measure_iceberg_indices(merged_df, soc_mappings, potential_threshold)
    
    summary = measure_summary(state_name, merged_df, industry_risk, total_econ_value, 
                             industry_risk["potential_econ_value_at_risk"].sum(), 
                             industry_risk["current_econ_value_at_risk"].sum(), 
                             potential_threshold)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        # Save merged_df with occupation-level data
        merged_df.to_csv(os.path.join(output_dir, f"{state_name.lower().replace(' ', '_')}_results.csv"), index=False)
        industry_risk.to_csv(os.path.join(output_dir, f"{state_name.lower().replace(' ', '_')}_industry_analysis.csv"), index=False)
        with open(os.path.join(output_dir, f"{state_name.lower().replace(' ', '_')}_summary.json"), "w") as f:
            json_safe_summary = {
                k: int(v) if isinstance(v, (np.int64, np.int32)) else float(v) if isinstance(v, (np.float64, np.float32)) else v
                for k, v in summary.items()
            }
            json.dump(json_safe_summary, f)
        
        # Save detailed occupation-level data
        occupation_details = merged_df[[
            "minor_group", "minor_group_name", "SOC_Code_Cleaned", "OCC_TITLE",
            "TOT_EMP", "A_MEAN", "economic_value", "automation_susceptibility",
            "is_potentially_at_risk", "potential_index", "enhanced_automation_risk",
            "is_currently_at_risk", "estimated_zapier_apps", "zapier_apps_per_worker"
        ]].sort_values(["minor_group", "SOC_Code_Cleaned"])
        occupation_details.to_csv(
            os.path.join(output_dir, f"{state_name.lower().replace(' ', '_')}_occupation_details.csv"),
            index=False
        )
        
        # Debug output for minor group 373
        if "373" in merged_df["minor_group"].values:
            print(f"\nDebugging Minor Group 373 for {state_name}:")
            print(merged_df[merged_df["minor_group"] == "373"][[
                "SOC_Code_Cleaned", "OCC_TITLE", "automation_susceptibility",
                "is_potentially_at_risk", "potential_index", "TOT_EMP", "economic_value"
            ]])
        
        visualize_industry_risk(state_name, industry_risk, output_dir)

    return merged_df, industry_risk, summary

def calculate_all_states_iceberg_indices(
    employment_file: str, tech_intensity_file: str, zapier_apps_file: str,
    soc_mapping_file: str, risk_threshold: int = 50, output_dir: Optional[str] = None, 
    sample_size: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate Iceberg Index for all states."""
    print("\n=== CALCULATING AUTOMATION RISK FOR ALL STATES ===")
    states = get_all_states(employment_file)
    if sample_size:
        states = states[:sample_size]
        print(f"Using sample of {sample_size} states.")

    summaries, industry_analyses, occupation_details_list = [], [], []
    failed_states = []
    for state_name in tqdm(states, desc="Processing states"):
        try:
            merged_df, industry_risk, summary = calculate_state_iceberg_index(
                state_name, employment_file, tech_intensity_file, zapier_apps_file, 
                soc_mapping_file, risk_threshold, output_dir
            )
            summaries.append(summary)
            industry_risk["state"] = state_name
            industry_analyses.append(industry_risk.copy())
            
            # Add state column to merged_df and collect for national summary
            merged_df["state"] = state_name
            occupation_details_list.append(merged_df.copy())
            
            print(f"Successfully processed {state_name}")
        except Exception as e:
            print(f"Error processing {state_name}: {e}")
            failed_states.append((state_name, str(e)))

    national_summary_df = pd.DataFrame(summaries)
    national_industry_df = pd.concat(industry_analyses, ignore_index=True)
    
    # Create national_summary_detailed by concatenating occupation-level data
    if occupation_details_list:
        national_summary_detailed = pd.concat(occupation_details_list, ignore_index=True)
        # Check for duplicates across states
        duplicates = national_summary_detailed[
            national_summary_detailed.duplicated(subset=["state", "SOC_Code_Cleaned"], keep=False)
        ]
        if not duplicates.empty:
            print(f"WARNING: Found {len(duplicates)} duplicate SOC codes in national_summary_detailed:")
            print(duplicates[["state", "SOC_Code_Cleaned", "OCC_TITLE", "TOT_EMP"]])
            national_summary_detailed = national_summary_detailed.drop_duplicates(subset=["state", "SOC_Code_Cleaned"])
            print(f"Removed duplicates. New national_summary_detailed size: {len(national_summary_detailed)}")
    else:
        print("WARNING: No occupation details collected. Creating empty national_summary_detailed.")
        national_summary_detailed = pd.DataFrame()

    if output_dir:
        national_summary_df.to_csv(os.path.join(output_dir, "national_summary.csv"), index=False)
        national_industry_df.to_csv(os.path.join(output_dir, "national_industry_analysis.csv"), index=False)
        national_summary_detailed.to_csv(os.path.join(output_dir, "national_summary_detailed.csv"), index=False)
        visualize_national_results(national_summary_df, national_industry_df, output_dir)

    return national_summary_df, national_industry_df

def get_all_states(employment_file: str) -> List[str]:
    """Get list of states from employment data."""
    emp_df = pd.read_csv(employment_file)
    states = [area for area in emp_df["AREA_TITLE"].unique() if
              area != "U.S." and not area.startswith("Metropolitan") and
              not area.startswith("Nonmetropolitan") and "," not in area]
    return sorted(states)

if __name__ == "__main__":
    print("\n=== PROJECT ICEBERG: CALCULATING AGENTIC API ECONOMY IMPACT ===")
    employment_file = "../data/state_M2023_dl-Table 1.csv"
    tech_intensity_file = "../tech_intensity_simple.csv"
    zapier_apps_file = "../data/zapier_apps_soc_codes.json"
    soc_mapping_file = "../data/soc_mapping.json"
    output_dir = "output_new2/"

    national_summary_df, national_industry_df = calculate_all_states_iceberg_indices(
        employment_file, tech_intensity_file, zapier_apps_file, soc_mapping_file,
        risk_threshold=50, output_dir=output_dir
    )
    print(f"\nProject Iceberg analysis complete. Results saved to: {output_dir}")
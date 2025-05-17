import os
import numpy as np
import pandas as pd
import json
from typing import Dict, List, Any, Optional

def generate_state_dashboard(output_dir: str = "output_new2/", dashboard_dir: str = "state_dashboards_v3/"):
    """
    Generate comprehensive state dashboard data from existing output files.
    Uses consistent metrics for automation risk assessment.
    
    Args:
        output_dir: Directory containing the state output files
        dashboard_dir: Directory to save the state dashboard files
    
    Returns:
        Dict mapping state names to their dashboard data
    """
    print(f"Generating state dashboards from {output_dir}")
    os.makedirs(dashboard_dir, exist_ok=True)
    
    # Get list of states from files in output directory
    state_files = {}
    for filename in os.listdir(output_dir):
        if "_results.csv" in filename:
            state_name = filename.replace("_results.csv", "").replace("_", " ").title()
            state_files[state_name] = {
                "results": os.path.join(output_dir, filename),
                "industry": os.path.join(output_dir, filename.replace("_results.csv", "_industry_analysis.csv")),
                "summary": os.path.join(output_dir, filename.replace("_results.csv", "_summary.json"))
            }
    
    # Load national summary for comparisons
    national_summary_file = os.path.join(output_dir, "national_summary.csv")
    if os.path.exists(national_summary_file):
        national_summary = pd.read_csv(national_summary_file)
    else:
        national_summary = None
    
    # Process each state
    state_dashboards = {}
    for state_name, files in state_files.items():
        try:
            print(f"Processing {state_name}...")
            
            # Load state data files
            results_file = files["results"]
            industry_file = files["industry"]
            summary_file = files["summary"]
            
            if not os.path.exists(results_file):
                print(f"  - Results file not found for {state_name}")
                continue
                
            merged_df = pd.read_csv(results_file)
            
            # Load industry analysis if available
            if os.path.exists(industry_file):
                industry_risk = pd.read_csv(industry_file)
                
                # Ensure automation_gap is correctly calculated
                if "automation_gap" not in industry_risk.columns and "minor_potential_index" in industry_risk.columns and "minor_iceberg_index" in industry_risk.columns:
                    industry_risk["automation_gap"] = industry_risk["minor_potential_index"] - industry_risk["minor_iceberg_index"]
                
                # Calculate econ_potential_index for industry if it doesn't exist
                if "econ_potential_index" not in industry_risk.columns:
                    if "minor_potential_index" in industry_risk.columns and "economic_value" in industry_risk.columns:
                        total_value = industry_risk["economic_value"].sum()
                        if total_value > 0:
                            industry_risk["econ_potential_index"] = industry_risk["minor_potential_index"] * (industry_risk["economic_value"] / total_value)
                            max_val = industry_risk["econ_potential_index"].max()
                            if max_val > 0:
                                industry_risk["econ_potential_index"] = (industry_risk["econ_potential_index"] / max_val) * 100
                        else:
                            industry_risk["econ_potential_index"] = industry_risk["minor_potential_index"]
            else:
                industry_risk = None
            
            # Load summary if available
            if os.path.exists(summary_file):
                with open(summary_file, "r") as f:
                    summary = json.load(f)
            else:
                summary = None
            
            # Generate dashboard data with consistent metrics
            dashboard = {
                # 1. Executive Summary
                "executive_summary": compute_executive_summary(state_name, merged_df, industry_risk, summary, national_summary),
                
                # 2. Economic Impact
                "economic_impact": compute_economic_impact(merged_df, industry_risk, summary),
                
                # 3. Workforce Impact
                "workforce_impact": compute_workforce_impact(merged_df, industry_risk, summary),
                
                # 4. Practical Action Items
                "action_items": compute_action_items(merged_df, industry_risk),
                
                # 5. Future Projections
                "future_projections": compute_future_projections(merged_df, summary),
                
                # 6. Comparative Analysis
                "comparative_analysis": compute_comparative_analysis(state_name, summary, national_summary)
            }
            
            # Save to file
            dashboard_file = os.path.join(dashboard_dir, f"{state_name.lower().replace(' ', '_')}_dashboard.json")
            with open(dashboard_file, "w") as f:
                # Convert numpy values to regular Python types
                def np_encoder(obj):
                    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
                        return int(obj)
                    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                        return float(obj)
                    elif isinstance(obj, (np.ndarray,)):
                        return obj.tolist()
                    return obj
                
                json.dump(dashboard, f, default=np_encoder, indent=2)
            
            state_dashboards[state_name] = dashboard
            print(f"  - Dashboard created for {state_name}")
            
        except Exception as e:
            print(f"  - Error processing {state_name}: {e}")
    
    # Generate national overview dashboard
    if state_dashboards:
        generate_national_overview(state_dashboards, national_summary, dashboard_dir)
    
    return state_dashboards


def compute_executive_summary(state_name: str, merged_df: pd.DataFrame, 
                             industry_risk: pd.DataFrame, summary: Dict, 
                             national_summary: pd.DataFrame) -> Dict:
    """
    Compute executive summary metrics.
    """
    # Get key metrics from summary if available
    if summary:
        iceberg_index = summary.get("iceberg_ratio", summary.get("pct_current_economic_value_at_risk", 0))
        potential_index = summary.get("potential_ratio", summary.get("pct_potential_economic_value_at_risk", 0))
        automation_gap = summary.get("automation_gap", potential_index - iceberg_index)
        economic_value_at_risk = summary.get("current_economic_value_at_risk", 0)
        employment_at_risk = summary.get("current_employment_at_risk", 0)
        total_employment = summary.get("total_employment", 0)
        total_econ_value = summary.get("economic_value_total", 0)
    else:
        # Calculate from merged_df
        iceberg_index = 0
        potential_index = 0
        automation_gap = 0
        economic_value_at_risk = 0
        employment_at_risk = 0
        total_employment = merged_df["TOT_EMP"].sum() if "TOT_EMP" in merged_df.columns else 0
        total_econ_value = merged_df["economic_value"].sum() if "economic_value" in merged_df.columns else 0
        
        # Calculate iceberg_index if we have the data
        if "enhanced_automation_risk" in merged_df.columns and "economic_value" in merged_df.columns:
            # Weight by economic value
            iceberg_index = (merged_df["enhanced_automation_risk"] * merged_df["economic_value"]).sum() / total_econ_value if total_econ_value > 0 else 0
        
        # Calculate potential_index if we have the data
        if "automation_susceptibility" in merged_df.columns and "economic_value" in merged_df.columns:
            potential_index = (merged_df["automation_susceptibility"] * merged_df["economic_value"]).sum() / total_econ_value if total_econ_value > 0 else 0
        
        automation_gap = potential_index - iceberg_index
        
        # Calculate economic value at risk
        if "is_currently_at_risk" in merged_df.columns and "economic_value" in merged_df.columns:
            economic_value_at_risk = merged_df[merged_df["is_currently_at_risk"]]["economic_value"].sum()
        
        # Calculate employment at risk
        if "is_currently_at_risk" in merged_df.columns and "TOT_EMP" in merged_df.columns:
            employment_at_risk = merged_df[merged_df["is_currently_at_risk"]]["TOT_EMP"].sum()
    
    # Calculate state ranking if national_summary is available
    ranking = "N/A"
    percentile = "N/A"
    if national_summary is not None:
        # Find ranking based on iceberg_index
        rank_col = None
        for col in ["iceberg_ratio", "pct_current_economic_value_at_risk"]:
            if col in national_summary.columns:
                rank_col = col
                break
        
        if rank_col:
            # Sort by risk index in descending order
            ranked_states = national_summary.sort_values(by=rank_col, ascending=False)
            state_row = ranked_states[ranked_states["state"] == state_name]
            
            if not state_row.empty:
                rank = ranked_states.index.get_loc(state_row.index[0]) + 1
                total_states = len(ranked_states)
                ranking = f"{rank} of {total_states}"
                percentile = 100 - (rank / total_states * 100)
    
    # Find top industries at risk - use minor_iceberg_index
    top_industries = []
    if industry_risk is not None and "minor_group_name" in industry_risk.columns:
        if "minor_iceberg_index" in industry_risk.columns:
            top_risk = industry_risk.sort_values("minor_iceberg_index", ascending=False).head(3)
            top_industries = top_risk["minor_group_name"].tolist()
    
    # Executive summary text
    exec_summary_text = (
        f"{state_name} has an Iceberg Index of {iceberg_index:.2f}%, with a potential automation "
        f"risk of {potential_index:.2f}%. This represents an economic value of ${economic_value_at_risk/1e9:.2f} "
        f"billion at current risk, affecting {employment_at_risk/1000:.1f}k workers. "
    )
    
    if top_industries:
        exec_summary_text += f"The highest-risk industries are {', '.join(top_industries)}."
    
    return {
        "state_name": state_name,
        "iceberg_index": iceberg_index,  # Current risk (%)
        "potential_index": potential_index,  # Potential risk (%)
        "automation_gap": automation_gap,  # Gap between potential and current (percentage points)
        "economic_value_at_risk": economic_value_at_risk,  # Current economic value at risk ($)
        "employment_at_risk": employment_at_risk,  # Current jobs at risk (count)
        "total_employment": total_employment,  # Total employment (count)
        "total_economic_value": total_econ_value,  # Total economic value ($)
        "state_ranking": ranking,  # State rank by Iceberg Index
        "state_percentile": percentile,  # State percentile by Iceberg Index
        "top_industries_at_risk": top_industries,  # Industries with highest current risk
        "summary_text": exec_summary_text,  # Summary text
        "metric_descriptions": {
            "iceberg_index": "Current percentage of economic value at risk of automation",
            "potential_index": "Theoretical percentage of economic value susceptible to automation",
            "automation_gap": "Difference between potential and current automation risk (percentage points)"
        }
    }


def compute_economic_impact(merged_df: pd.DataFrame, industry_risk: pd.DataFrame, summary: Dict) -> Dict:
    """
    Compute economic impact metrics.
    """
    # Top industries by economic value at risk, using minor_iceberg_index
    top_industries = []
    if industry_risk is not None and "minor_group_name" in industry_risk.columns:
        sort_col = "minor_potential_index"  # Prioritize current risk for economic impact
        if sort_col in industry_risk.columns:
            sorted_industries = industry_risk.sort_values(sort_col, ascending=False).head(10)
            
            for _, row in sorted_industries.iterrows():
                industry = {
                    "name": row["minor_group_name"],
                    "code": row["minor_group"],
                    "economic_value_at_risk": row.get("potential_econ_value_at_risk", 0),
                    "percentage_at_risk": row.get(sort_col, 0),
                    # Use minor_potential_index as fallback for econ_potential_index
                    "minor_potential_index": row.get("minor_potential_index", 0),
                    "employment": row.get("TOT_EMP", 0)
                }
                top_industries.append(industry)
    
    # GDP impact by risk level
    gdp_by_risk = {}
    if "automation_risk_category" in merged_df.columns and "economic_value" in merged_df.columns:
        risk_groups = merged_df.groupby("automation_risk_category")["economic_value"].sum()
        total_econ = risk_groups.sum()
        
        for category, value in risk_groups.items():
            gdp_by_risk[category] = {
                "economic_value": value,
                "percentage": (value / total_econ * 100) if total_econ > 0 else 0
            }
    
    # Calculate total economic value at risk
    total_at_risk = 0
    percentage_at_risk = 0
    if summary:
        total_at_risk = summary.get("current_economic_value_at_risk", 0)
        percentage_at_risk = summary.get("pct_current_economic_value_at_risk", summary.get("iceberg_ratio", 0))
    elif "is_currently_at_risk" in merged_df.columns and "economic_value" in merged_df.columns:
        total_at_risk = merged_df[merged_df["is_currently_at_risk"]]["economic_value"].sum()
        total_econ = merged_df["economic_value"].sum()
        percentage_at_risk = (total_at_risk / total_econ * 100) if total_econ > 0 else 0
    
    return {
        "top_industries_by_value": top_industries,  # Industries with highest current risk
        "gdp_by_risk_level": gdp_by_risk,  # Economic value grouped by risk category
        "total_economic_value_at_risk": total_at_risk,  # Current economic value at risk ($)
        "percentage_of_economy_at_risk": percentage_at_risk,  # Current percentage of economy at risk
        "metric_descriptions": {
            "percentage_at_risk": "Current percentage of industry's economic value at risk (minor_iceberg_index)",
            "econ_potential_index": "Potential automation impact weighted by economic importance",
            "economic_value_at_risk": "Current economic value at risk in dollars"
        }
    }


def compute_workforce_impact(merged_df: pd.DataFrame, industry_risk: pd.DataFrame, summary: Dict) -> Dict:
    """
    Compute workforce impact metrics with proper employment-weighted risk calculations.
    """
    # Calculate total employment for weighting
    total_employment = merged_df["TOT_EMP"].sum() if "TOT_EMP" in merged_df.columns else 0
    
    # Calculate workforce-weighted potential and current risk
    total_potential_workforce_risk = 0
    total_current_workforce_risk = 0
    
    if total_employment > 0 and "TOT_EMP" in merged_df.columns:
        # Calculate potential risk (weighted by employment)
        if "automation_susceptibility" in merged_df.columns:
            merged_df["potential_workforce_risk"] = merged_df["automation_susceptibility"] * merged_df["TOT_EMP"] / total_employment
            total_potential_workforce_risk = merged_df["potential_workforce_risk"].sum()
        
        # Calculate current risk (weighted by employment)
        if "enhanced_automation_risk" in merged_df.columns:
            merged_df["current_workforce_risk"] = merged_df["enhanced_automation_risk"] * merged_df["TOT_EMP"] / total_employment
            total_current_workforce_risk = merged_df["current_workforce_risk"].sum()
    
    # Top occupations by potential risk
    top_potential_occupations = []
    if "automation_susceptibility" in merged_df.columns and "TOT_EMP" in merged_df.columns and "OCC_TITLE" in merged_df.columns:
        # Sort by automation_susceptibility (potential risk)
        sorted_by_potential = merged_df.sort_values(["automation_susceptibility", "TOT_EMP"], ascending=[False, False]).head(10)
        
        for _, row in sorted_by_potential.iterrows():
            occupation = {
                "title": row["OCC_TITLE"],
                "soc_code": row.get("SOC_Code_Cleaned", ""),
                "employment": row["TOT_EMP"],
                "mean_wage": row.get("A_MEAN", 0),
                "potential_risk": row["automation_susceptibility"],
                "economic_value": row.get("economic_value", 0),
                "weighted_potential_risk": row.get("potential_workforce_risk", 0)
            }
            top_potential_occupations.append(occupation)
    
    # Top occupations by current risk
    top_current_occupations = []
    if "enhanced_automation_risk" in merged_df.columns and "TOT_EMP" in merged_df.columns and "OCC_TITLE" in merged_df.columns:
        # Sort by enhanced_automation_risk (current risk)
        sorted_by_current = merged_df.sort_values(["enhanced_automation_risk", "TOT_EMP"], ascending=[False, False]).head(10)
        
        for _, row in sorted_by_current.iterrows():
            occupation = {
                "title": row["OCC_TITLE"],
                "soc_code": row.get("SOC_Code_Cleaned", ""),
                "employment": row["TOT_EMP"],
                "mean_wage": row.get("A_MEAN", 0),
                "current_risk": row["enhanced_automation_risk"],
                "economic_value": row.get("economic_value", 0),
                "weighted_current_risk": row.get("current_workforce_risk", 0)
            }
            top_current_occupations.append(occupation)
    
    # Top occupations by automation gap
    top_gap_occupations = []
    if all(col in merged_df.columns for col in ["automation_susceptibility", "enhanced_automation_risk", "OCC_TITLE", "TOT_EMP"]):
        # Calculate automation gap
        merged_df["automation_gap"] = merged_df["automation_susceptibility"] - merged_df["enhanced_automation_risk"]
        
        # Sort by automation gap
        sorted_by_gap = merged_df.sort_values(["automation_gap", "TOT_EMP"], ascending=[False, False]).head(10)
        
        for _, row in sorted_by_gap.iterrows():
            occupation = {
                "title": row["OCC_TITLE"],
                "soc_code": row.get("SOC_Code_Cleaned", ""),
                "employment": row["TOT_EMP"],
                "mean_wage": row.get("A_MEAN", 0),
                "potential_risk": row["automation_susceptibility"],
                "current_risk": row["enhanced_automation_risk"],
                "automation_gap": row["automation_gap"],
                "economic_value": row.get("economic_value", 0)
            }
            top_gap_occupations.append(occupation)
    
    # Workforce concentration by wage level
    workforce_by_wage = {}
    if all(col in merged_df.columns for col in ["A_MEAN", "TOT_EMP"]):
        # Split by wage quartiles as a simple proxy for different regions
        merged_df["wage_quartile"] = pd.qcut(
            merged_df["A_MEAN"], 
            4, 
            labels=["Lower Wage", "Lower-Middle Wage", "Upper-Middle Wage", "Higher Wage"]
        )
        
        wage_groups = merged_df.groupby("wage_quartile")
        
        for quartile, group in wage_groups:
            group_total_emp = group["TOT_EMP"].sum()
            
            # Calculate metrics for potential risk
            potential_at_risk_emp = 0
            group_potential_risk = 0
            if "automation_susceptibility" in merged_df.columns:
                threshold = 60  # Threshold for considering a job "at risk"
                potential_at_risk_emp = group[group["automation_susceptibility"] > threshold]["TOT_EMP"].sum()
                
                if group_total_emp > 0:
                    group_potential_risk = (group["automation_susceptibility"] * group["TOT_EMP"]).sum() / group_total_emp
            
            # Calculate metrics for current risk
            current_at_risk_emp = 0
            group_current_risk = 0
            if "enhanced_automation_risk" in merged_df.columns:
                current_at_risk_emp = group[group["enhanced_automation_risk"] > threshold]["TOT_EMP"].sum()
                
                if group_total_emp > 0:
                    group_current_risk = (group["enhanced_automation_risk"] * group["TOT_EMP"]).sum() / group_total_emp
            
            workforce_by_wage[quartile] = {
                "total_employment": group_total_emp,
                "potential_employment_at_risk": potential_at_risk_emp,
                "current_employment_at_risk": current_at_risk_emp,
                "potential_risk_percentage": (potential_at_risk_emp / group_total_emp * 100) if group_total_emp > 0 else 0,
                "current_risk_percentage": (current_at_risk_emp / group_total_emp * 100) if group_total_emp > 0 else 0,
                "weighted_potential_risk": group_potential_risk,
                "weighted_current_risk": group_current_risk,
                "automation_gap": group_potential_risk - group_current_risk,
                "average_wage": group["A_MEAN"].mean()
            }
    
    # Calculate overall workforce metrics
    total_employment = 0
    potential_at_risk = 0
    current_at_risk = 0
    potential_percentage = 0
    current_percentage = 0
    
    if summary:
        total_employment = summary.get("total_employment", 0)
        # Use potential values if available
        potential_at_risk = summary.get("potential_employment_at_risk", 0)
        potential_percentage = summary.get("pct_potential_employment_at_risk", 0)
        # Use current values
        current_at_risk = summary.get("current_employment_at_risk", 0)
        current_percentage = summary.get("pct_current_employment_at_risk", 0)
    elif "TOT_EMP" in merged_df.columns:
        total_employment = merged_df["TOT_EMP"].sum()
        # Calculate potential risk
        if "automation_susceptibility" in merged_df.columns:
            threshold = 60  # Threshold for considering a job "at risk"
            potential_at_risk = merged_df[merged_df["automation_susceptibility"] > threshold]["TOT_EMP"].sum()
            potential_percentage = (potential_at_risk / total_employment * 100) if total_employment > 0 else 0
        # Calculate current risk
        if "enhanced_automation_risk" in merged_df.columns:
            current_at_risk = merged_df[merged_df["enhanced_automation_risk"] > threshold]["TOT_EMP"].sum()
            current_percentage = (current_at_risk / total_employment * 100) if total_employment > 0 else 0
    
    # Potential risk distribution by category
    potential_risk_distribution = {}
    if "automation_susceptibility" in merged_df.columns and "TOT_EMP" in merged_df.columns:
        merged_df["potential_risk_category"] = merged_df["automation_susceptibility"].apply(
            lambda score: "Very High" if score >= 80 else 
                         "High" if score >= 60 else
                         "Moderate" if score >= 40 else
                         "Low" if score >= 20 else "Very Low"
        )
        risk_groups = merged_df.groupby("potential_risk_category")["TOT_EMP"].sum()
        
        for category, emp_count in risk_groups.items():
            potential_risk_distribution[category] = {
                "employment": emp_count,
                "percentage": (emp_count / total_employment * 100) if total_employment > 0 else 0
            }
    
    # Current risk distribution by category
    current_risk_distribution = {}
    if "enhanced_automation_risk" in merged_df.columns and "TOT_EMP" in merged_df.columns:
        merged_df["current_risk_category"] = merged_df["enhanced_automation_risk"].apply(
            lambda score: "Very High" if score >= 80 else 
                         "High" if score >= 60 else
                         "Moderate" if score >= 40 else
                         "Low" if score >= 20 else "Very Low"
        )
        risk_groups = merged_df.groupby("current_risk_category")["TOT_EMP"].sum()
        
        for category, emp_count in risk_groups.items():
            current_risk_distribution[category] = {
                "employment": emp_count,
                "percentage": (emp_count / total_employment * 100) if total_employment > 0 else 0
            }
    
    return {
        "top_occupations_by_potential_risk": top_potential_occupations,
        "top_occupations_by_current_risk": top_current_occupations,
        "top_occupations_by_automation_gap": top_gap_occupations,
        "workforce_by_wage_level": workforce_by_wage,
        "total_employment": total_employment,
        "potential_employment_at_risk": potential_at_risk,
        "current_employment_at_risk": current_at_risk,
        "potential_percentage_at_risk": potential_percentage,
        "current_percentage_at_risk": current_percentage,
        "total_potential_workforce_risk": total_potential_workforce_risk,
        "total_current_workforce_risk": total_current_workforce_risk,
        "workforce_automation_gap": total_potential_workforce_risk - total_current_workforce_risk,
        "potential_risk_distribution": potential_risk_distribution,
        "current_risk_distribution": current_risk_distribution,
        "metric_descriptions": {
            "automation_susceptibility": "Theoretical automation susceptibility based on job tasks",
            "enhanced_automation_risk": "Current automation risk incorporating both susceptibility and tool availability",
            "potential_workforce_risk": "Potential risk weighted by employment proportion (automation_susceptibility * job_emp / total_emp)",
            "current_workforce_risk": "Current risk weighted by employment proportion (enhanced_automation_risk * job_emp / total_emp)",
            "automation_gap": "Difference between potential and current automation risk (percentage points)"
        }
    }


def compute_action_items(merged_df: pd.DataFrame, industry_risk: pd.DataFrame) -> Dict:
    """
    Compute practical action items.
    """
    # Industries with largest automation gaps
    gap_industries = []
    if industry_risk is not None and "minor_group_name" in industry_risk.columns:
        # Ensure automation_gap is calculated correctly
        if "automation_gap" not in industry_risk.columns and "minor_potential_index" in industry_risk.columns and "minor_iceberg_index" in industry_risk.columns:
            industry_risk["automation_gap"] = industry_risk["minor_potential_index"] - industry_risk["minor_iceberg_index"]
        
        if "automation_gap" in industry_risk.columns:
            sorted_gaps = industry_risk.sort_values("automation_gap", ascending=False).head(10)
            
            for _, row in sorted_gaps.iterrows():
                industry = {
                    "name": row["minor_group_name"],
                    "code": row["minor_group"],
                    "current_risk": row["minor_iceberg_index"] if "minor_iceberg_index" in row else 0,  # Current risk
                    "potential_risk": row["minor_potential_index"] if "minor_potential_index" in row else 0,  # Potential risk
                    "automation_gap": row["automation_gap"],  # Gap between potential and current
                    "employment": row["TOT_EMP"] if "TOT_EMP" in row else 0,  # Employment
                    # Use minor_potential_index as fallback for econ_potential_index
                    "econ_potential_index": row.get("econ_potential_index", row.get("minor_potential_index", 0))  # Economic importance
                }
                gap_industries.append(industry)
    
    # Occupations by automation gap
    gap_occupations = []
    if "OCC_TITLE" in merged_df.columns:
        # Use automation_susceptibility for potential and enhanced_automation_risk for current
        if "automation_susceptibility" in merged_df.columns and "enhanced_automation_risk" in merged_df.columns:
            # Calculate automation gap
            merged_df["occupation_gap"] = merged_df["econ_potential_index"]
            # merged_df["automation_susceptibility"] - merged_df["enhanced_automation_risk"]
            
            # Sort by gap and economic value
            sorted_by_gap = merged_df.sort_values(["occupation_gap", "economic_value"], ascending=[False, False]).head(10)
            
            for _, row in sorted_by_gap.iterrows():
                occupation = {
                    "title": row["OCC_TITLE"],
                    "soc_code": row["SOC_Code_Cleaned"],
                    "current_risk": row["enhanced_automation_risk"],  # Current risk
                    "potential_risk": row["automation_susceptibility"],  # Potential risk
                    "automation_gap": row["occupation_gap"],  # Gap between potential and current
                    "employment": row["TOT_EMP"] if "TOT_EMP" in row else 0,  # Employment
                    "mean_wage": row["A_MEAN"] if "A_MEAN" in row else 0,  # Mean wage
                    "economic_value": row["economic_value"] if "economic_value" in row else 0  # Economic value
                }
                gap_occupations.append(occupation)
    
    # MCP opportunities (where MCP servers could help)
    mcp_opportunities = []
    if "OCC_TITLE" in merged_df.columns and "zapier_apps_per_worker" in merged_df.columns:
        # Find occupations with low MCP coverage but high automation potential
        if "automation_susceptibility" in merged_df.columns:
            # Normalize zapier_apps_per_worker to 0-100 scale for comparison
            max_apps = merged_df["zapier_apps_per_worker"].max()
            if max_apps > 0:
                merged_df["mcp_coverage_norm"] = merged_df["zapier_apps_per_worker"] / max_apps * 100
            else:
                merged_df["mcp_coverage_norm"] = 0
            
            # Calculate MCP opportunity as high potential but low coverage
            merged_df["mcp_opportunity"] = merged_df["automation_susceptibility"] - merged_df["mcp_coverage_norm"]
            
            # Get top opportunities
            sorted_opps = merged_df.sort_values("mcp_opportunity", ascending=False).head(10)
            
            for _, row in sorted_opps.iterrows():
                opportunity = {
                    "title": row["OCC_TITLE"],
                    "soc_code": row["SOC_Code_Cleaned"],
                    "automation_potential": row["automation_susceptibility"],  # Theoretical potential
                    "current_mcp_coverage": row["mcp_coverage_norm"],  # Current MCP coverage
                    "mcp_opportunity": row["mcp_opportunity"],  # Gap between potential and coverage
                    "employment": row["TOT_EMP"] if "TOT_EMP" in row else 0,  # Employment
                    "economic_value": row["economic_value"] if "economic_value" in row else 0  # Economic value
                }
                mcp_opportunities.append(opportunity)
    
    # Retraining recommendations - use enhanced_automation_risk to identify high-risk jobs
    retraining_suggestions = []
    if "OCC_TITLE" in merged_df.columns and "A_MEAN" in merged_df.columns:
        risk_col = "econ_potential_index"  # Use current risk for retraining recommendations
        
        if risk_col in merged_df.columns:
            # Find high-risk jobs
            merged_df_sorted = merged_df.sort_values(risk_col, ascending=False)
            high_risk_jobs = merged_df_sorted.head(int(len(merged_df) * 0.25))  # Top 25% are high risk
            low_risk_jobs = merged_df_sorted.tail(int(len(merged_df) * 0.25))   # Bottom 25% are low risk
            
            if not high_risk_jobs.empty and not low_risk_jobs.empty:
                # Get top 5 jobs at highest risk
                top_risk_jobs = high_risk_jobs.head(5)
                
                for _, risk_job in top_risk_jobs.iterrows():
                    risk_wage = risk_job["A_MEAN"]
                    
                    # Find jobs with similar wages but lower risk
                    wage_diff = 0.2  # Look for jobs within 20% wage difference
                    similar_wage_jobs = low_risk_jobs[
                        (low_risk_jobs["A_MEAN"] >= risk_wage * (1 - wage_diff)) &
                        (low_risk_jobs["A_MEAN"] <= risk_wage * (1 + wage_diff))
                    ]
                    
                    alternatives = []
                    for _, alt_job in similar_wage_jobs.head(3).iterrows():
                        alternatives.append({
                            "title": alt_job["OCC_TITLE"],
                            "mean_wage": alt_job["A_MEAN"],
                            "risk_score": alt_job[risk_col],
                            "risk_metric": "Current Automation Risk"
                        })
                    
                    retraining_suggestions.append({
                        "at_risk_job": {
                            "title": risk_job["OCC_TITLE"],
                            "mean_wage": risk_job["A_MEAN"],
                            "risk_score": risk_job[risk_col],
                            "risk_metric": "Current Automation Risk",
                            "employment": risk_job.get("TOT_EMP", 0)
                        },
                        "alternative_jobs": alternatives
                    })
    
    return {
        "industries_with_largest_gaps": gap_industries,  # Industries with largest automation gaps
        "occupations_with_largest_gaps": gap_occupations,  # Occupations with largest automation gaps
        "mcp_server_opportunities": mcp_opportunities,  # Occupations with MCP opportunity
        "retraining_suggestions": retraining_suggestions,  # Retraining recommendations
        "metric_descriptions": {
            "current_risk": "Current automation risk (minor_iceberg_index for industries, enhanced_automation_risk for occupations)",
            "potential_risk": "Theoretical automation potential (minor_potential_index for industries, automation_susceptibility for occupations)",
            "automation_gap": "Difference between potential and current automation risk (percentage points)",
            "mcp_opportunity": "Gap between automation potential and current MCP server coverage (percentage points)"
        }
    }


def compute_future_projections(merged_df: pd.DataFrame, summary: Dict) -> Dict:
    """
    Compute future projections.
    """
    # Get baseline metrics
    current_iceberg = 0
    potential_index = 0
    automation_gap = 0
    total_econ_value = 0
    total_employment = 0
    current_econ_at_risk = 0
    current_emp_at_risk = 0
    
    if summary:
        current_iceberg = summary.get("iceberg_ratio", summary.get("pct_current_economic_value_at_risk", 0))
        potential_index = summary.get("potential_ratio", summary.get("pct_potential_economic_value_at_risk", 0))
        automation_gap = summary.get("automation_gap", potential_index - current_iceberg)
        total_econ_value = summary.get("economic_value_total", 0)
        total_employment = summary.get("total_employment", 0)
        current_econ_at_risk = summary.get("current_economic_value_at_risk", 0)
        current_emp_at_risk = summary.get("current_employment_at_risk", 0)
    else:
        # Calculate from merged_df if we have it
        if "economic_value" in merged_df.columns:
            total_econ_value = merged_df["economic_value"].sum()
            
            if "is_currently_at_risk" in merged_df.columns:
                current_econ_at_risk = merged_df[merged_df["is_currently_at_risk"]]["economic_value"].sum()
                current_iceberg = (current_econ_at_risk / total_econ_value * 100) if total_econ_value > 0 else 0
            
            if "enhanced_automation_risk" in merged_df.columns:
                current_iceberg = (merged_df["enhanced_automation_risk"] * merged_df["economic_value"]).sum() / total_econ_value if total_econ_value > 0 else 0
            
            if "automation_susceptibility" in merged_df.columns:
                potential_index = (merged_df["automation_susceptibility"] * merged_df["economic_value"]).sum() / total_econ_value if total_econ_value > 0 else 0
                
        if "TOT_EMP" in merged_df.columns:
            total_employment = merged_df["TOT_EMP"].sum()
            
            if "is_currently_at_risk" in merged_df.columns:
                current_emp_at_risk = merged_df[merged_df["is_currently_at_risk"]]["TOT_EMP"].sum()
    
    automation_gap = potential_index - current_iceberg
    
    # Projection assumptions
    # 1. Short-term (1 year): 50% of the automation gap closes
    # 2. Medium-term (3 years): 75% of the automation gap closes
    # 3. Long-term (5 years): 90% of the automation gap closes
    
    short_term_iceberg = current_iceberg + (automation_gap * 0.5)
    medium_term_iceberg = current_iceberg + (automation_gap * 0.75)
    long_term_iceberg = current_iceberg + (automation_gap * 0.9)
    
    # Economic impact
    short_term_econ = (short_term_iceberg / 100) * total_econ_value
    medium_term_econ = (medium_term_iceberg / 100) * total_econ_value
    long_term_econ = (long_term_iceberg / 100) * total_econ_value
    
    # Employment impact
    short_term_emp = (short_term_iceberg / current_iceberg) * current_emp_at_risk if current_iceberg > 0 else (short_term_iceberg / 100) * total_employment
    medium_term_emp = (medium_term_iceberg / current_iceberg) * current_emp_at_risk if current_iceberg > 0 else (medium_term_iceberg / 100) * total_employment
    long_term_emp = (long_term_iceberg / current_iceberg) * current_emp_at_risk if current_iceberg > 0 else (long_term_iceberg / 100) * total_employment
    
    return {
        "current": {
            "iceberg_index": current_iceberg,  # Current automation risk (%)
            "potential_index": potential_index,  # Potential automation risk (%)
            "automation_gap": automation_gap,  # Gap between potential and current (percentage points)
            "economic_value_at_risk": current_econ_at_risk,  # Current economic value at risk ($)
            "employment_at_risk": current_emp_at_risk  # Current jobs at risk (count)
        },
        "short_term": {
            "timeframe": "12 months",
            "iceberg_index": short_term_iceberg,  # Projected automation risk in 12 months (%)
            "economic_value_at_risk": short_term_econ,  # Projected economic value at risk in 12 months ($)
            "employment_at_risk": short_term_emp,  # Projected jobs at risk in 12 months (count)
            "gap_closure": "50%"  # Percentage of automation gap closed in 12 months
        },
        "medium_term": {
            "timeframe": "3 years",
            "iceberg_index": medium_term_iceberg,  # Projected automation risk in 3 years (%)
            "economic_value_at_risk": medium_term_econ,  # Projected economic value at risk in 3 years ($)
            "employment_at_risk": medium_term_emp,  # Projected jobs at risk in 3 years (count)
            "gap_closure": "75%"  # Percentage of automation gap closed in 3 years
        },
        "long_term": {
            "timeframe": "5 years",
            "iceberg_index": long_term_iceberg,  # Projected automation risk in 5 years (%)
            "economic_value_at_risk": long_term_econ,  # Projected economic value at risk in 5 years ($)
            "employment_at_risk": long_term_emp,  # Projected jobs at risk in 5 years (count)
            "gap_closure": "90%"  # Percentage of automation gap closed in 5 years
        },
        "assumptions": [
            "Projections assume continuous development and adoption of MCP servers",
            "Short-term projection assumes 50% of automation gap is closed within 1 year",
            "Medium-term projection assumes 75% of automation gap is closed within 3 years",
            "Long-term projection assumes 90% of automation gap is closed within 5 years"
        ],
        "metric_descriptions": {
            "iceberg_index": "Percentage of economic value at risk of automation",
            "economic_value_at_risk": "Economic value at risk in dollars",
            "employment_at_risk": "Number of jobs at risk"
        }
    }

def compute_comparative_analysis(state_name: str, summary: Dict, national_summary: pd.DataFrame) -> Dict:
    """
    Compute comparative analysis metrics.
    """
    # Get state metrics
    state_iceberg = 0
    state_potential = 0
    state_automation_gap = 0
    
    if summary:
        state_iceberg = summary.get("iceberg_ratio", summary.get("pct_current_economic_value_at_risk", 0))
        state_potential = summary.get("potential_ratio", summary.get("pct_potential_economic_value_at_risk", 0))
        state_automation_gap = summary.get("automation_gap", state_potential - state_iceberg)
    
    # National comparisons
    national_iceberg = 1.75  # Default from Project Iceberg document
    national_potential = 12.6  # Default from Project Iceberg document
    national_automation_gap = national_potential - national_iceberg  # Default from Project Iceberg document
    
    # Get actual national averages if available
    if national_summary is not None:
        # Find the appropriate columns
        iceberg_col = None
        potential_col = None
        
        for col in ["iceberg_ratio", "pct_current_economic_value_at_risk"]:
            if col in national_summary.columns:
                iceberg_col = col
                break
                
        for col in ["potential_ratio", "pct_potential_economic_value_at_risk"]:
            if col in national_summary.columns:
                potential_col = col
                break
        
        if iceberg_col:
            national_iceberg = national_summary[iceberg_col].mean()
            
        if potential_col:
            national_potential = national_summary[potential_col].mean()
            
        if iceberg_col and potential_col:
            national_automation_gap = national_potential - national_iceberg
    
    # Find similar states
    similar_states = []
    if national_summary is not None and "state" in national_summary.columns:
        # Find states with similar Iceberg Indices
        iceberg_col = None
        for col in ["iceberg_ratio", "pct_current_economic_value_at_risk"]:
            if col in national_summary.columns:
                iceberg_col = col
                break
                
        if iceberg_col:
            # Find the state's iceberg index in national_summary
            state_row = national_summary[national_summary["state"] == state_name]
            if not state_row.empty:
                state_index = state_row[iceberg_col].iloc[0]
                
                # Find states with similar indices (within ±25%)
                similar_mask = (
                    (national_summary["state"] != state_name) & 
                    (national_summary[iceberg_col] >= state_index * 0.75) & 
                    (national_summary[iceberg_col] <= state_index * 1.25)
                )
                
                # Get up to 5 similar states
                similar_states_df = national_summary[similar_mask].sort_values(
                    by=iceberg_col, 
                    key=lambda x: abs(x - state_index)
                ).head(5)
                
                for _, row in similar_states_df.iterrows():
                    similar_states.append({
                        "state": row["state"],
                        "iceberg_index": row[iceberg_col],
                        "difference": row[iceberg_col] - state_index
                    })
    
    # Determine neighboring states (this would require a map of state borders)
    # For now, use a simplified approach based on state names
    neighboring_states = []
    
    # Compute state trend (whether risk is increasing/decreasing)
    # This would require time-series data
    trend = "stable"  # Default
    
    return {
        "national_comparison": {
            "state_iceberg_index": state_iceberg,  # Current automation risk for state (%)
            "national_iceberg_index": national_iceberg,  # National average current automation risk (%)
            "iceberg_differential": state_iceberg - national_iceberg,  # Difference between state and national (percentage points)
            "iceberg_comparison": "above" if state_iceberg > national_iceberg else "below",  # Whether state is above or below national average
            
            "state_potential_index": state_potential,  # Potential automation risk for state (%)
            "national_potential_index": national_potential,  # National average potential automation risk (%)
            "potential_differential": state_potential - national_potential,  # Difference between state and national (percentage points)
            "potential_comparison": "above" if state_potential > national_potential else "below",  # Whether state is above or below national average
            
            "state_automation_gap": state_automation_gap,  # Automation gap for state (percentage points)
            "national_automation_gap": national_automation_gap,  # National average automation gap (percentage points)
            "gap_differential": state_automation_gap - national_automation_gap,  # Difference between state and national (percentage points)
            "gap_comparison": "above" if state_automation_gap > national_automation_gap else "below"  # Whether state is above or below national average
        },
        "similar_states": similar_states,  # States with similar Iceberg Indices
        "neighboring_states": neighboring_states,  # Neighboring states (placeholder)
        "trend": trend,  # Trend in automation risk (placeholder)
        "metric_descriptions": {
            "iceberg_index": "Current percentage of economic value at risk of automation",
            "potential_index": "Theoretical percentage of economic value susceptible to automation",
            "automation_gap": "Difference between potential and current automation risk (percentage points)",
            "differential": "Difference between state and national average (percentage points)"
        }
    }

def generate_national_overview(state_dashboards: Dict, national_summary: pd.DataFrame, dashboard_dir: str):
    """
    Generate a national overview dashboard comparing all states.
    """
    # Extract key metrics from each state dashboard
    states_comparison = []
    for state_name, dashboard in state_dashboards.items():
        if "executive_summary" in dashboard:
            summary = dashboard["executive_summary"]
            states_comparison.append({
                "state": state_name,
                "iceberg_index": summary.get("iceberg_index", 0),  # Current automation risk (%)
                "potential_index": summary.get("potential_index", 0),  # Potential automation risk (%)
                "automation_gap": summary.get("automation_gap", 0),  # Gap between potential and current (percentage points)
                "economic_value_at_risk": summary.get("economic_value_at_risk", 0),  # Current economic value at risk ($)
                "employment_at_risk": summary.get("employment_at_risk", 0),  # Current jobs at risk (count)
                "total_employment": summary.get("total_employment", 0),  # Total employment (count)
                "total_economic_value": summary.get("total_economic_value", 0)  # Total economic value ($)
            })
    
    # Convert to DataFrame for easier analysis
    if states_comparison:
        states_df = pd.DataFrame(states_comparison)
        
        # Calculate national averages
        national_avg = {
            "iceberg_index": states_df["iceberg_index"].mean(),  # National average current risk (%)
            "potential_index": states_df["potential_index"].mean(),  # National average potential risk (%)
            "automation_gap": states_df["automation_gap"].mean(),  # National average automation gap (percentage points)
            "total_economic_value_at_risk": states_df["economic_value_at_risk"].sum(),  # Total economic value at risk ($)
            "total_employment_at_risk": states_df["employment_at_risk"].sum(),  # Total jobs at risk (count)
            "total_economic_value": states_df["total_economic_value"].sum(),  # Total economic value ($)
            "total_employment": states_df["total_employment"].sum(),  # Total employment (count)
            "percentage_of_economy_at_risk": (states_df["economic_value_at_risk"].sum() / states_df["total_economic_value"].sum() * 100) 
                if states_df["total_economic_value"].sum() > 0 else 0,  # Percentage of economy at risk (%)
            "percentage_of_workforce_at_risk": (states_df["employment_at_risk"].sum() / states_df["total_employment"].sum() * 100)
                if states_df["total_employment"].sum() > 0 else 0  # Percentage of workforce at risk (%)
        }
        
        # Get top and bottom states
        top_states = states_df.sort_values("iceberg_index", ascending=False).head(5)
        bottom_states = states_df.sort_values("iceberg_index").head(5)
        
        top_states_list = [{"state": row["state"], "iceberg_index": row["iceberg_index"]} for _, row in top_states.iterrows()]
        bottom_states_list = [{"state": row["state"], "iceberg_index": row["iceberg_index"]} for _, row in bottom_states.iterrows()]
        
        # States with largest automation gaps
        gap_states = states_df.sort_values("automation_gap", ascending=False).head(5)
        gap_states_list = [{"state": row["state"], "automation_gap": row["automation_gap"]} for _, row in gap_states.iterrows()]
        
        # Create national overview
        national_overview = {
            "national_averages": national_avg,
            "states_highest_risk": top_states_list,
            "states_lowest_risk": bottom_states_list,
            "states_largest_gaps": gap_states_list,
            "all_states": states_comparison,
            "metric_descriptions": {
                "iceberg_index": "Current percentage of economic value at risk of automation",
                "potential_index": "Theoretical percentage of economic value susceptible to automation",
                "automation_gap": "Difference between potential and current automation risk (percentage points)",
                "economic_value_at_risk": "Current economic value at risk in dollars",
                "employment_at_risk": "Current number of jobs at risk"
            }
        }
        
        # Save to file
        overview_file = os.path.join(dashboard_dir, "national_overview.json")
        with open(overview_file, "w") as f:
            # Convert numpy values to regular Python types
            def np_encoder(obj):
                if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
                    return int(obj)
                elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.ndarray,)):
                    return obj.tolist()
                return obj
            
            json.dump(national_overview, f, default=np_encoder, indent=2)
        
        print(f"National overview dashboard created and saved to {overview_file}")
    else:
        print("No state dashboards available to generate national overview")

if __name__ == "__main__":
    # Generate state dashboards from output_new2 directory
    state_dashboards = generate_state_dashboard(
        output_dir="output_new2/", 
        dashboard_dir="state_dashboards_v3/"
    )
    
    print(f"Successfully generated dashboards for {len(state_dashboards)} states")
    
    # Print metrics used for reference
    print("\nMetrics used in dashboard:")
    print("1. Current Risk:")
    print("   - Industry level: minor_iceberg_index (percentage of industry's economic value at risk)")
    print("   - Occupation level: enhanced_automation_risk (weighted combination of susceptibility and MCP coverage)")
    print("   - State level: iceberg_ratio (percentage of state's economic value at risk)")
    
    print("\n2. Potential Risk:")
    print("   - Industry level: minor_potential_index (theoretical percentage of industry's economic value at risk)")
    print("   - Occupation level: automation_susceptibility (theoretical automation susceptibility)")
    print("   - State level: potential_ratio (theoretical percentage of state's economic value at risk)")
    
    print("\n3. Automation Gap:")
    print("   - Direct calculation: potential_risk - current_risk (percentage points)")
    print("   - Used to identify automation opportunities and future projections")
    
    print("\n4. Economic Metrics:")
    print("   - economic_value_at_risk: current economic value at risk in dollars")
    print("   - economic_value: total economic value (employment × wages)")
    print("   - econ_potential_index: economic value-weighted automation potential")
    
    print("\n5. Employment Metrics:")
    print("   - employment_at_risk: number of jobs currently at risk")
    print("   - total_employment: total employment count")
    print("   - percentage_at_risk: percentage of jobs at risk")
    
    print("\nAll metrics are consistently documented with descriptions in the dashboard JSON.")
import pandas as pd
import numpy as np
import os

OPPORTUNITY_JOBS_OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data_v2/opportunity_jobs.csv")
# SKILLS_IMPORTANCE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/skills/skills_importance.csv")
SKILLS_BASED_RISK_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/skills/skills_based_risk.csv")

def add_technical_composition(df_normalized_skill_importance):
    """
    Add technical composition column to the dataframe
    """
    print(df_normalized_skill_importance.columns)
    df_normalized_skill_importance['technical_composition'] = df_normalized_skill_importance['technical_skills'] / (df_normalized_skill_importance['cognitive_skills'] + df_normalized_skill_importance['social_skills'] + df_normalized_skill_importance['operations_skills'] + df_normalized_skill_importance['maintenance_skills'] + df_normalized_skill_importance['technical_skills'] + df_normalized_skill_importance['management_skills'])
    return df_normalized_skill_importance

def analyze_technical_skills_percentiles(df_normalized_skill_importance):
    """
    Read CSV file and calculate 80th and 20th percentiles for technical_skills column
    
    Args:
        csv_path (str): Path to the CSV file
    
    Returns:
        dict: Dictionary containing the percentile values
    """
    try:
        # Read the CSV file
        df = df_normalized_skill_importance
        
        # Check if technical_skills column exists
        if 'technical_skills' not in df.columns:
            raise ValueError("Column 'technical_skills' not found in the CSV file")
        
        # Calculate percentiles
        percentile_20 = np.percentile(df['technical_composition'], 20)
        percentile_80 = np.percentile(df['technical_composition'], 80)
        
        # Additional statistics for context
        mean_value = df['technical_composition'].mean()
        median_value = df['technical_composition'].median()
        std_value = df['technical_composition'].std()
        min_value = df['technical_composition'].min()
        max_value = df['technical_composition'].max()
        
        results = {
            '20th_percentile': percentile_20,
            '80th_percentile': percentile_80,
            'mean': mean_value,
            'median': median_value,
            'std_deviation': std_value,
            'min': min_value,
            'max': max_value,
            'total_records': len(df),
            'new_df': df
        }
        
        return results
        
    except FileNotFoundError:
        print(f"Error: File '{csv_path}' not found.")
        return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def find_opportunity_jobs(df: pd.DataFrame, version: str = "v1") -> pd.DataFrame:
    """
    Find opportunity jobs based on skills based risk and skills importance
    """
    if version == "v1":
        percentile_20 = 0.1165
        percentile_80 = 0.1558
        df_filtered = df[df['technical_composition'] >= percentile_20]
        df_filtered = df_filtered[df_filtered['technical_composition'] <= percentile_80]
        df_filtered.sort_values(by='technical_composition', ascending=False, inplace=True)
        return df_filtered[['O*NET-SOC Code', 'Title', 'technical_composition', 'cognitive_skills', 'social_skills', 'operations_skills', 'maintenance_skills', 'management_skills', 'technical_skills']]
    elif version == "v2":
        # Add AI benefit classification
        df_filtered = df.copy()
        
        def classify_ai_benefit_potential(row):
            """Classify jobs based on AI benefit potential using key features"""
            ops_skills = row['operations_skills']
            cog_skills = row['cognitive_skills'] 
            soc_skills = row['social_skills']
            tech_comp = row['technical_composition']
            mgmt_skills = row['management_skills']
            
            score = 0
            
            # Primary factors
            if cog_skills >= 3.3:
                score += 2
            elif cog_skills >= 3.0:
                score += 1
                
            if ops_skills <= 1.5:
                score += 2
            elif ops_skills <= 2.0:
                score += 1
            elif ops_skills >= 2.5:
                score -= 2
                
            # Secondary factors
            if tech_comp >= 0.14:
                score += 1
            if 2.8 <= soc_skills <= 3.4:
                score += 1
            elif soc_skills >= 3.6:
                score -= 1
            if mgmt_skills >= 2.5:
                score += 1
                
            # Final classification
            if score >= 4:
                return "High AI Benefit"
            elif score >= 2:
                return "Moderate AI Benefit"  
            elif score >= 0:
                return "Low AI Benefit"
            else:
                return "Minimal AI Benefit"
        
        def calculate_ai_benefit_score(row):
            """Calculate numerical AI benefit score for ranking"""
            ops_skills = row['operations_skills']
            cog_skills = row['cognitive_skills'] 
            soc_skills = row['social_skills']
            tech_comp = row['technical_composition']
            mgmt_skills = row['management_skills']
            
            score = 0
            
            # Primary factors (stronger weights)
            score += cog_skills * 0.3  # Higher cognitive = better
            score -= ops_skills * 0.25  # Lower operations = better
            
            # Secondary factors
            score += tech_comp * 2.0    # Technical orientation
            score += mgmt_skills * 0.1  # Management capability
            
            # Social skills - moderate range is optimal
            if 2.8 <= soc_skills <= 3.4:
                score += soc_skills * 0.05
            elif soc_skills >= 3.6:
                score -= (soc_skills - 3.6) * 0.1  # Penalty for very high social dependency
            
            return score
        
        df_filtered['AI_Benefit_Classification'] = df_filtered.apply(classify_ai_benefit_potential, axis=1)
        df_filtered['AI_Benefit_Score'] = df_filtered.apply(calculate_ai_benefit_score, axis=1)
        
        # Sort by AI benefit score (highest first)
        df_filtered.sort_values(by='AI_Benefit_Score', ascending=False, inplace=True)
        
        return df_filtered[['O*NET-SOC Code', 'Title', 'technical_composition', 'cognitive_skills', 'social_skills', 
                           'operations_skills', 'maintenance_skills', 'management_skills', 'technical_skills',
                           'AI_Benefit_Classification', 'AI_Benefit_Score']]

def main():
    # Analyze the technical skills percentiles
    df_normalized_skill_importance = pd.read_csv(SKILLS_BASED_RISK_PATH)
    df_normalized_skill_importance = add_technical_composition(df_normalized_skill_importance)
    results = analyze_technical_skills_percentiles(df_normalized_skill_importance)
    
    if results:
        print("Technical Skills Composition Analysis Results:")
        print("=" * 40)
        print(f"20th Percentile: {results['20th_percentile']:.4f}")
        print(f"80th Percentile: {results['80th_percentile']:.4f}")
        print("\nAdditional Statistics:")
        print(f"Mean: {results['mean']:.4f}")
        print(f"Median: {results['median']:.4f}")
        print(f"Standard Deviation: {results['std_deviation']:.4f}")
        print(f"Minimum: {results['min']:.4f}")
        print(f"Maximum: {results['max']:.4f}")
        print(f"Total Records: {results['total_records']}")
        
        # Show the range between percentiles
        percentile_range = results['80th_percentile'] - results['20th_percentile']
        print(f"\nRange between 80th and 20th percentiles: {percentile_range:.4f}")

    # Generate opportunity jobs - you can change version to "v2" for AI classification
    version = "v2"  # Change to "v2" for AI benefit analysis
    print(f"\nGenerating opportunity jobs using version: {version}")
    
    df_opportunity_jobs = find_opportunity_jobs(
        df_normalized_skill_importance, 
        version=version
    )
    
    # Update output path based on version
    if version == "v2":
        output_path = OPPORTUNITY_JOBS_OUTPUT_PATH.replace(".csv", "_with_ai_classification.csv")
        print(f"\n=== AI BENEFIT CLASSIFICATION RESULTS ===")
        if 'AI_Benefit_Classification' in df_opportunity_jobs.columns:
            classification_counts = df_opportunity_jobs['AI_Benefit_Classification'].value_counts()
            total_jobs = len(df_opportunity_jobs)
            for category, count in classification_counts.items():
                percentage = (count / total_jobs) * 100
                print(f"{category}: {count} jobs ({percentage:.1f}%)")
    else:
        output_path = OPPORTUNITY_JOBS_OUTPUT_PATH
    
    df_opportunity_jobs.to_csv(output_path, index=False)
    print(f"\nOpportunity jobs saved to {output_path}")
    
    # Show sample of top results
    print(f"\nTop 10 opportunity jobs:")
    if version == "v2":
        print("Ranked by AI Benefit Score:")
        for i, (_, row) in enumerate(df_opportunity_jobs.head(10).iterrows(), 1):
            print(f"{i:2d}. {row['Title']} (AI: {row['AI_Benefit_Classification']}, Score: {row['AI_Benefit_Score']:.3f})")
    else:
        print("Ranked by Technical Composition:")
        for i, (_, row) in enumerate(df_opportunity_jobs.head(10).iterrows(), 1):
            print(f"{i:2d}. {row['Title']} (Tech Comp: {row['technical_composition']:.3f})")

if __name__ == "__main__":
    main()

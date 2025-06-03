import pandas as pd
import numpy as np
import os

OPPORTUNITY_JOBS_OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data_v2/opportunity_jobs_{version}.csv")
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
    percentile_20 = 0.1165
    percentile_80 = 0.1558
    if version == "v1":
        df_filtered = df[df['technical_composition'] >= percentile_20]
        df_filtered = df_filtered[df_filtered['technical_composition'] <= percentile_80]
        df_filtered.sort_values(by='technical_composition', ascending=False, inplace=True)
        return df_filtered[['O*NET-SOC Code', 'Title', 'technical_composition', 'cognitive_skills', 'social_skills', 'operations_skills', 'maintenance_skills', 'management_skills', 'technical_skills']]
    elif version == "v2":
        # Comprehensive AI Opportunity Analysis across all technical composition levels
        df_filtered = df.copy()
        
        # Define technical composition strata
        def get_tech_composition_stratum(tech_comp):
            if tech_comp <= percentile_20:
                return "Low-Tech"
            elif tech_comp <= percentile_80:
                return "Medium-Tech" 
            else:
                return "High-Tech"
        
        def classify_ai_transformation_potential(row):
            """Classify LOW-TECH/MEDIUM-TECH jobs based on their potential to be transformed by AI tools"""
            ops_skills = row['operations_skills']
            cog_skills = row['cognitive_skills'] 
            soc_skills = row['social_skills']
            tech_comp = row['technical_composition']
            mgmt_skills = row['management_skills']
            title = row['Title']
            
            # Only apply to low-tech jobs
            if tech_comp > percentile_80:
                return "Not Applicable"
            
            score = 0
            
        
            # But has cognitive capacity to adopt AI tools
            if cog_skills >= 3.0:
                score += 2
            elif cog_skills >= 2.8:
                score += 1
                
            # Jobs involving information processing, decision making, customer interaction
            if 2.5 <= soc_skills <= 3.8:  # Human interaction that AI can augment
                score += 1
                
            # if mgmt_skills >= 2.2:  # Management/coordination tasks AI can enhance
            #     score += 1
                
            # Lower operations skills = less physical, more suitable for AI augmentation
            if ops_skills <= 2.0:
                score += 2
            elif ops_skills <= 2.5:
                score += 1
                
                    
            # Penalty for purely manual/physical jobs
            if ops_skills >= 3.0:
                score -= 2
                
            # Final classification
            if score >= 4:
                return "High Transformation Potential"
            elif score >= 2:
                return "Moderate Transformation Potential"
            elif score >= 0:
                return "Low Transformation Potential"
            else:
                return "Minimal Transformation Potential"
        
        def classify_ai_enhancement_potential(row):
            """Classify HIGH-TECH jobs based on their potential to benefit from advanced AI"""
            ops_skills = row['operations_skills']
            cog_skills = row['cognitive_skills'] 
            soc_skills = row['social_skills']
            tech_comp = row['technical_composition']
            mgmt_skills = row['management_skills']
            tech_skills = row['technical_skills']
            title = row['Title']
            
            if tech_comp <= percentile_80:
                return "Not Applicable"
            
            score = 0
            
            # Primary factors for AI enhancement in technical roles
            score += cog_skills * 0.25  # Cognitive skills important for advanced AI adoption
            score -= ops_skills * 0.3   # Physical work reduces AI benefit potential
            score += tech_comp * 2.0    # Higher tech composition = more AI enhancement potential
            score += tech_skills * 0.15 # Technical skills help with AI adoption
            
            # # Management roles can benefit significantly from AI insights
            # if mgmt_skills >= 2.5:
            #     score += 0.4
            # elif mgmt_skills >= 2.0:
            #     score += 0.2
                
            # Final classification  
            if score >= 2.0:
                return "High Enhancement Potential"
            elif score >= 1.5:
                return "Moderate Enhancement Potential"
            elif score >= 1.0:
                return "Low Enhancement Potential"
            else:
                return "Minimal Enhancement Potential"
        
        def calculate_transformation_score(row):
            """Calculate score for LOW-TECH transformation potential"""
            if row['technical_composition'] > 0.13:
                return 0
                
            ops_skills = row['operations_skills']
            cog_skills = row['cognitive_skills'] 
            soc_skills = row['social_skills']
            tech_comp = row['technical_composition']
            mgmt_skills = row['management_skills']
            title = row['Title']
            
            score = 0
            
            # Core transformation factors
            score += (0.13 - tech_comp) * 10  # Higher score for lower current tech
            score += cog_skills * 0.4  # Cognitive ability to adopt new tools
            
            # Human-centric jobs that AI can augment
            if 2.5 <= soc_skills <= 3.8:
                score += soc_skills * 0.3
                
            # Management/coordination roles
            score += mgmt_skills * 0.25
            
            # Less physical = more suitable for AI tools
            score -= ops_skills * 0.2
            
            return max(0, score)
        
        def calculate_enhancement_score(row):
            """Calculate score for MEDIUM/HIGH-TECH enhancement potential"""
            if row['technical_composition'] <= 0.13:
                return 0
                
            ops_skills = row['operations_skills']
            cog_skills = row['cognitive_skills'] 
            soc_skills = row['social_skills']
            tech_comp = row['technical_composition']
            mgmt_skills = row['management_skills']
            tech_skills = row['technical_skills']
            
            score = 0
            
            # Base scoring for technical roles
            score += cog_skills * 0.25
            score -= ops_skills * 0.3  
            score += tech_comp * 2.0
            score += tech_skills * 0.15
            
            # Management bonus
            if mgmt_skills >= 2.5:
                score += 0.4
            elif mgmt_skills >= 2.0:
                score += 0.2
                
            
            return max(0, score)
        
        # Apply all classifications
        df_filtered['Tech_Stratum'] = df_filtered['technical_composition'].apply(get_tech_composition_stratum)
        df_filtered['AI_Transformation_Classification'] = df_filtered.apply(classify_ai_transformation_potential, axis=1)
        df_filtered['AI_Enhancement_Classification'] = df_filtered.apply(classify_ai_enhancement_potential, axis=1)

                
        return df_filtered[['O*NET-SOC Code', 'Title', 'technical_composition', 'Tech_Stratum',
                           'cognitive_skills', 'social_skills', 'operations_skills', 'maintenance_skills', 
                           'management_skills', 'technical_skills', 'AI_Transformation_Classification',
                           'AI_Enhancement_Classification']]

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

    output_path = OPPORTUNITY_JOBS_OUTPUT_PATH.format(version=version)
    
    df_opportunity_jobs = find_opportunity_jobs(
        df_normalized_skill_importance, 
        version=version
    ) 
    df_opportunity_jobs.to_csv(output_path, index=False)
    print(f"\nOpportunity jobs saved to {output_path}")

if __name__ == "__main__":
    main()

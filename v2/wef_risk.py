import numpy as np
import pandas as pd
import os
import re
import requests

SKILL_IMPORTANCE_PATH = os.path.join(os.path.dirname(__file__), "data", "skills", "skills_importance.csv")

class WefRisk:
    def __init__(self):
        self.skills_df = pd.read_csv(SKILL_IMPORTANCE_PATH)
        self.skill_mapping = {
            'basic_skills': [
                'Reading Comprehension', 'Active Listening', 'Writing', 'Speaking',
            ],
            'cognitive_skills': [
                'Critical Thinking', 'Active Learning', 'Learning Strategies',
                'Monitoring', 'Complex Problem Solving', 'Judgment and Decision Making', 'Operations Analysis'
            ],
            'social_skills': [
                'Social Perceptiveness', 'Coordination', 'Persuasion',
                'Negotiation', 'Instructing', 'Service Orientation'
            ],
            'operations_skills': [
                'Operation and Control', 'Operations Monitoring', 'Quality Control Analysis', 'Troubleshooting'
            ],
            'maintenance_skills': [
                'Equipment Selection', 'Installation', 'Equipment Maintenance', 'Repairing'
            ],
            'technical_skills': [
                'Technology Design', 'Programming', 'Mathematics', 'Science'
            ],
            'management_skills': [
                'Systems Analysis', 'Systems Evaluation', 'Time Management',
                'Management of Financial Resources', 'Management of Material Resources',
                'Management of Personnel Resources'
            ]
        }
        self.categorized_df = self.get_categorized_df()
        # Pattern for identifying growth sectors (care, education, tech, green, management)
        self.GROWTH_SECTOR_PATTERNS = re.compile(
            r"(?:nurs|therap|counsel|teacher|educat|"      # care & education
            r"ai\b|ml\b|machine learning|data|cyber|"    # digital / AI / security
            r"engineer|developer|analyst|"               # generic digital titles
            r"renewable|solar|wind|green|sustain|"       # green transition
            r"project manager|operations manager)",      # leadership / project
            flags=re.I
        )
        

    def get_categorized_df(self):
        """
            Generate and return the categorized dataframe.
            Categorized dataframe combines similar skills as mentioned in the skill_mapping dictionary into 7 skill categories.
            This is done by computing the mean of the mapped skills for each category.
        """
        skill_importance_df = pd.read_csv(SKILL_IMPORTANCE_PATH)
        categorized_data_dict = {
            'O*NET-SOC Code': skill_importance_df['O*NET-SOC Code'],
            'Title': skill_importance_df['Title'],
        }
        for cat, skills in self.skill_mapping.items():
            present_skills = [s for s in skills if s in skill_importance_df.columns]
            categorized_data_dict[cat] = skill_importance_df[present_skills].mean(axis=1)
        self.categorized_df = pd.DataFrame(categorized_data_dict)
        return self.categorized_df

    def _norm(self, series: pd.Series) -> pd.Series:
        return (series - 1.0) / 4.0

    def compute_risk_score(self, onet_soc_code) -> float:
        """
        Compute the risk score for a given job title and skill importance ratings.

        Args:
            onet_soc_code (str): The SOC code of the job title. SOC code of format XX-XXXX.XX

        Returns:
            float: The risk score for the job title. Score is between 0 and 100.
        """
        df = self.categorized_df
        df = df[df["O*NET-SOC Code"] == onet_soc_code]
        if df.empty:
            raise ValueError(f"No data found for SOC code: {onet_soc_code}")
        
        # Step 1: Normalize the skill values
        basic_skills_value  = self._norm(df["basic_skills"])
        cognitive_skills_value  = self._norm(df["cognitive_skills"])
        social_skills_value  = self._norm(df["social_skills"])
        operations_skills_value  = self._norm(df["operations_skills"])
        maintenance_skills_value  = self._norm(df["maintenance_skills"])
        technical_skills_value  = self._norm(df["technical_skills"])
        management_skills_value = self._norm(df["management_skills"])


        # Step 2: Compute 3 core indices
        routine_intensity = 0.5 * operations_skills_value + 0.5 * maintenance_skills_value
        
        human_capital     = (
            0.40 * social_skills_value + 
            0.30 * management_skills_value + 
            0.20 * cognitive_skills_value + 
            0.10 * basic_skills_value
        )   # ↑ social/management weight, ↓ basic
        
        tech_shield       = 0.30 * technical_skills_value + 0.10 * cognitive_skills_value

        # 3. raw risk (0‑1) with updated weights
        raw = (
            0.55 * routine_intensity
            + 0.30 * (1 - human_capital)
            + 0.15 * (1 - tech_shield)
        )

        # 4. WEF growth‑sector discount (‑30 %) – stricter threshold
        mask_growth = df["Title"].str.contains(self.GROWTH_SECTOR_PATTERNS, na=False) & (
            (social_skills_value >= 0.60) | (technical_skills_value >= 0.60)   # need ≥3.4 in raw 1‑5 scale
        )
        raw = raw.mask(mask_growth, raw * 0.70)

        # Step 3: Scale to 0‑100
        df = df.copy()
        df["automation_risk"] = (raw * 100).round(1)
        df["automation_risk_score"] = (
            100 / (1 + np.exp(-4 * (raw - 0.40)))
        ).round(1)

        return df["automation_risk_score"].values[0]

if __name__ == "__main__":
    wef_risk = WefRisk()
    print(wef_risk.compute_risk_score("17-2072.00"))
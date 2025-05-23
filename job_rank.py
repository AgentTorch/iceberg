import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from sklearn.preprocessing import StandardScaler
import json
import os
import argparse
import pickle

# File paths
DATA_DIR = 'data'
ONET_DIR = os.path.join(DATA_DIR, 'ONET')
STATEWISE_DATA_DIR = os.path.join('output_new2')
STORE_DIR = 'store'
JOBRANK_DIR = os.path.join(STORE_DIR, 'jobrank')


# Input files
SKILLS_CSV = os.path.join(ONET_DIR, 'Skills.csv')
TECH_INTENSITY_CSV = os.path.join(DATA_DIR, 'tech_intensity_simple.csv')
SOC_MAPPING_JSON = os.path.join(DATA_DIR, 'soc_mapping.json')

# Output files
SKILL_EMBEDDINGS_NPY = os.path.join(JOBRANK_DIR, 'skill_embeddings.npy')
SIMILARITY_MATRIX_NPY = os.path.join(JOBRANK_DIR, 'similarity_matrix.npy')
TRANSITION_MATRIX_NPY = os.path.join(JOBRANK_DIR, 'transition_matrix.npy')
SKILL_EMBEDDINGS_META_PKL = os.path.join(JOBRANK_DIR, 'skill_embeddings_meta.pkl')
JOB_GRAPH_GPICKLE = os.path.join(JOBRANK_DIR, 'job_graph.gpickle')
JOB_RANK_SCORES_PKL = os.path.join(JOBRANK_DIR, 'job_rank_scores.pkl')
ENHANCED_AUTOMATION_RISK_CSV = os.path.join(STATEWISE_DATA_DIR, '{state}_occupation_details.csv')

def get_minor_code(soc_code):
    """
        Return the minor code from the SOC code in accordance to our format of minor codes
    """

    c = soc_code.replace('-', '')
    return c[:3]

def get_soc_code_cleaned(soc_code):
    """
        Return the SOC code cleaned in accordance to our format of SOC codes
    """
    c = soc_code.split('.')[0]
    c = c.replace('-', '')
    return c


class JobRank:
    def __init__(self, state=None, load_data=False):
        self.state = state
        self.skills_df = None
        self.tech_intensity_df = None
        self.soc_mapping = None
        self.graph = nx.DiGraph()
        # aux data
        self.skill_embeddings = None
        self.similarity_matrix = None
        self.transition_matrix = None
        self.job_rank_scores = None
        self.enhanced_automation_risk_df = None
        
        
        
        self.skill_categories = {
            'basic_skills': [
                'Reading Comprehension', 'Active Listening', 'Writing', 'Speaking',
                'Mathematics', 'Science'
            ],
            'cognitive_skills': [
                'Critical Thinking', 'Active Learning', 'Learning Strategies',
                'Monitoring', 'Complex Problem Solving', 'Judgment and Decision Making'
            ],
            'social_skills': [
                'Social Perceptiveness', 'Coordination', 'Persuasion',
                'Negotiation', 'Instructing', 'Service Orientation'
            ],
            'technical_skills': [
                'Technology Design', 'Programming', 'Operations Analysis',
                'Equipment Selection', 'Installation', 'Operations Monitoring',
                'Operation and Control', 'Equipment Maintenance', 'Troubleshooting',
                'Repairing', 'Quality Control Analysis'
            ],
            'management_skills': [
                'Systems Analysis', 'Systems Evaluation', 'Time Management',
                'Management of Financial Resources', 'Management of Material Resources',
                'Management of Personnel Resources'
            ]
        }

        if load_data:
            self.load_data()
        
    def save_data(self, directory=JOBRANK_DIR):
        """Save all JobRank data to files"""
        os.makedirs(directory, exist_ok=True)
        
        # Save numpy arrays
        np.save(SKILL_EMBEDDINGS_NPY, self.skill_embeddings.values)
        np.save(SIMILARITY_MATRIX_NPY, self.similarity_matrix.values)
        np.save(TRANSITION_MATRIX_NPY, self.transition_matrix.values)
        
        # Save indices and columns
        with open(SKILL_EMBEDDINGS_META_PKL, 'wb') as f:
            pickle.dump({
                'index': self.skill_embeddings.index.tolist(),
                'columns': self.skill_embeddings.columns.tolist()
            }, f)
        
        # Save graph
        with open(JOB_GRAPH_GPICKLE, 'wb') as f:
            pickle.dump(self.graph, f)
        
        # Save job rank scores
        with open(JOB_RANK_SCORES_PKL, 'wb') as f:
            pickle.dump(self.job_rank_scores, f)
        
        print(f"Saved JobRank data to {directory}")
    
    def load_data(self, directory=JOBRANK_DIR):
        """Load all JobRank data from files"""
        self.load_input_data()
        # Load numpy arrays
        self.skill_embeddings = pd.DataFrame(
            np.load(SKILL_EMBEDDINGS_NPY),
            **pickle.load(open(SKILL_EMBEDDINGS_META_PKL, 'rb'))
        )
        print(f'Loaded skill embeddings: {self.skill_embeddings.shape}')
        self.similarity_matrix = pd.DataFrame(
            np.load(SIMILARITY_MATRIX_NPY),
            index=self.skill_embeddings.index,
            columns=self.skill_embeddings.index
        )
        print(f'Loaded similarity matrix: {self.similarity_matrix.shape}')
        self.transition_matrix = pd.DataFrame(
            np.load(TRANSITION_MATRIX_NPY),
            index=self.skill_embeddings.index,
            columns=self.skill_embeddings.index
        )
        print(f'Loaded transition matrix: {self.transition_matrix.shape}')
        # Load graph
        with open(JOB_GRAPH_GPICKLE, 'rb') as f:
            self.graph = pickle.load(f)
            print(f'Loaded nodes: {len(self.graph.nodes())}')
        
        # Load job rank scores
        with open(JOB_RANK_SCORES_PKL, 'rb') as f:
            self.job_rank_scores = pickle.load(f)
        
        print(f"Loaded JobRank data from {directory}")
        
    def create_detailed_skill_embeddings(self):
        """Create detailed skill-based embeddings for each job"""
        # Filter for only Importance scale values
        importance_skills = self.skills_df[self.skills_df['Scale Name'] == 'Importance']
        
        # Create skill importance matrix
        skill_matrix = importance_skills.pivot(
            index='O*NET-SOC Code',
            columns='Element Name',
            values='Data Value'
        ).fillna(0)
        
        # Normalize the embeddings
        scaler = StandardScaler()
        self.skill_embeddings = pd.DataFrame(
            scaler.fit_transform(skill_matrix),
            index=skill_matrix.index,
            columns=skill_matrix.columns
        )
        
        return self.skill_embeddings
    
    def calculate_skill_transition_cost(self, source_skills, target_skills):
        """Calculate the cost of transitioning based on skill differences"""
        # Calculate skill gaps
        skill_gaps = target_skills - source_skills
        
        # Weight gaps by skill importance
        weighted_gaps = skill_gaps.abs() * target_skills
        
        # Calculate transition cost
        self.transition_cost = weighted_gaps.sum(axis=1)
        
        return self.transition_cost
    
    def calculate_skill_similarity(self):
        """Calculate similarity between jobs based on detailed skills"""
        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(self.skill_embeddings)
        
        # Convert to DataFrame
        self.similarity_matrix = pd.DataFrame(
            similarity_matrix,
            index=self.skill_embeddings.index,
            columns=self.skill_embeddings.index
        )
        
        return self.similarity_matrix
    
    def calculate_transition_probability(self):
        """Calculate transition probabilities considering detailed skills"""
        n_jobs = len(self.similarity_matrix)
        transition_matrix = np.zeros((n_jobs, n_jobs))
        
        for i, job1 in enumerate(self.similarity_matrix.index):
            for j, job2 in enumerate(self.similarity_matrix.columns):
                if i != j:
                    # Base similarity
                    sim = self.similarity_matrix.loc[job1, job2]
                    
                    # Skill transition cost
                    skill_cost = self.calculate_skill_transition_cost(
                        self.skill_embeddings.loc[job1:job1],
                        self.skill_embeddings.loc[job2:job2]
                    ).iloc[0]
                    
                    # Normalize skill cost to [0,1] range
                    skill_factor = 1 / (1 + skill_cost)
                    
                    # Automation risk factor
                    # TODO: utilize ICEBERG index here instead of tech_intensity_score which is just one of the many factors
                    risk1 = self.get_automation_risk(job1) # self.tech_intensity_df.loc[job1, 'tech_intensity_score']
                    risk2 = self.get_automation_risk(job2) # self.tech_intensity_df.loc[job2, 'tech_intensity_score']
                    if risk2 > risk1:
                        transition_matrix[i, j] = 0
                    else:
                        # Higher score when target job has lower automation risk
                        risk_factor = 1 - (risk2 / risk1) if risk1 > 0 else 0
                        
                        # Hot technology factor
                        # TODO: experiment removing this as this may not a helpful factor
                        hot_ratio1 = self.tech_intensity_df.loc[job1, 'hot_tech_ratio']
                        hot_ratio2 = self.tech_intensity_df.loc[job2, 'hot_tech_ratio']
                        hot_factor = 1 - abs(hot_ratio1 - hot_ratio2)
                        
                        # Combine all factors
                        transition_matrix[i, j] = (
                            sim * 0.4 +  # Base similarity
                            skill_factor * 0.3 +  # Skill transition cost
                            risk_factor * 0.2 +  # Automation risk
                            hot_factor * 0.1  # Hot technology
                        )
        
        # Normalize rows to get probabilities
        row_sums = transition_matrix.sum(axis=1)
        transition_matrix = transition_matrix / row_sums[:, np.newaxis]
        
        self.transition_matrix = pd.DataFrame(
            transition_matrix,
            index=self.similarity_matrix.index,
            columns=self.similarity_matrix.columns
        )
        
        return self.transition_matrix
    
    def build_job_graph(self):
        """Build directed graph of job transitions with skill information"""
        for i, job1 in enumerate(self.transition_matrix.index):
            for j, job2 in enumerate(self.transition_matrix.columns):
                if job1 == job2:
                    continue
                if self.transition_matrix.loc[job1, job2] > 0:
                    self.graph.add_edge(
                        job1,
                        job2,
                        weight=self.transition_matrix.loc[job1, job2]
                    )
    
    def calculate_job_rank(self, alpha=0.85, max_iter=100):
        """Calculate JobRank scores using PageRank algorithm"""
        self.job_rank_scores = {job: 1/len(self.graph) for job in self.graph.nodes()}
        
        for _ in range(max_iter):
            new_scores = {}
            for job in self.graph.nodes():
                incoming_score = sum(
                    self.job_rank_scores[predecessor] * self.graph[predecessor][job]['weight']
                    for predecessor in self.graph.predecessors(job)
                )
                new_scores[job] = (1 - alpha) + alpha * incoming_score
            
            total = sum(new_scores.values())
            self.job_rank_scores = {job: score/total for job, score in new_scores.items()}
        
        return self.job_rank_scores
    
    def get_transition_recommendations(self, job_code, n_recommendations=5):
        """Get detailed transition recommendations with skill analysis"""
        if job_code not in self.graph:
            print(f"Job code {job_code} not found in graph")
            return []
        
        # Get outgoing edges sorted by weight
        transitions = sorted(
            self.graph.edges(job_code, data=True),
            key=lambda x: x[2]['weight'],
            reverse=True
        )
        
        # Get source job skills
        source_skills = self.skill_embeddings.loc[job_code]
        
        recommendations = []
        for _, target, data in transitions[:n_recommendations]:
            # Get target job skills
            target_skills = self.skill_embeddings.loc[target]
            
            # Calculate skill gaps
            skill_gaps = self._calculate_skill_gaps(source_skills, target_skills)

            soc_mapping_target = self.soc_mapping.get(target, None)
            
            if soc_mapping_target is None:
                modified_target = target.split('.')[0]
                soc_mapping_target = self.soc_mapping.get(modified_target, {})
            
            if soc_mapping_target is None:
                print(f"No SOC mapping found for {target}")
                
            soc_mapping_target_title = soc_mapping_target.get('detailed_title', "")

            recommendations.append({
                'target_job': target,
                'target_title': soc_mapping_target_title,
                'transition_probability': data['weight'],
                'tech_intensity': self.tech_intensity_df.loc[target, 'tech_intensity_score'],
                'hot_tech_ratio': self.tech_intensity_df.loc[target, 'hot_tech_ratio'],
                'skill_gaps': skill_gaps,
                'required_training': self._identify_required_training(skill_gaps)
            })
        
        return recommendations
    
    def _calculate_skill_gaps(self, source_skills, target_skills):
        """Calculate specific skill gaps between jobs"""
        gaps = []
        # source_skills and target_skills are now Series with skill names as index
        for skill_name in target_skills.index:
            source_value = source_skills[skill_name]
            target_value = target_skills[skill_name]
            
            if target_value > 0 and source_value == 0:
                gaps.append({
                    'skill': skill_name,
                    'gap': target_value,
                    'type': 'new_skill'
                })
            elif target_value > source_value:
                gaps.append({
                    'skill': skill_name,
                    'gap': target_value - source_value,
                    'type': 'skill_upgrade'
                })
        
        return sorted(gaps, key=lambda x: x['gap'], reverse=True)
    
    def _identify_required_training(self, skill_gaps):
        """Identify required training based on skill gaps"""
        training_needs = []
        
        for gap in skill_gaps:
            if gap['type'] == 'new_skill':
                training_needs.append({
                    'skill': gap['skill'],
                    'training_type': 'comprehensive_training',
                    'estimated_duration': '3-6 months'
                })
            elif gap['gap'] > 0.5:  # Significant skill upgrade needed
                # TODO VERIFY THUS - This is purely LLM generated and needs to be verified
                training_needs.append({
                    'skill': gap['skill'],
                    'training_type': 'advanced_training',
                    'estimated_duration': '1-3 months'
                })
            else:
                training_needs.append({
                    'skill': gap['skill'],
                    'training_type': 'refresher_course',
                    'estimated_duration': '2-4 weeks'
                })
        
        return training_needs

    def load_input_data(self):
        """Load input data"""
        self.skills_df = pd.read_csv(SKILLS_CSV)
        self.tech_intensity_df = pd.read_csv(TECH_INTENSITY_CSV, index_col='O*NET-SOC Code')
        with open(SOC_MAPPING_JSON, 'r') as f:
            self.soc_mapping = json.load(f)

        enhanced_automation_risk_csv = ENHANCED_AUTOMATION_RISK_CSV.format(state=self.state)
        self.enhanced_automation_risk_df = pd.read_csv(enhanced_automation_risk_csv, index_col='SOC_Code_Cleaned')

    def get_automation_risk(self, soc_code):
        """
            Return the automation risk for a given SOC code.
            If SOC code is not found, returns the median automation risk.
        """
        soc_code_cleaned = get_soc_code_cleaned(soc_code)
        try:
            return self.enhanced_automation_risk_df.loc[int(soc_code_cleaned), 'enhanced_automation_risk']
        except KeyError:
            median_risk = self.enhanced_automation_risk_df['enhanced_automation_risk'].median()
            print(f"Warning: SOC code {soc_code_cleaned} not found in automation risk data. Using median risk: {median_risk:.3f}")
            return median_risk
        
    
    def build(self):
        self.load_input_data()
        """Run the complete JobRank processing pipeline"""
        print("-----> Creating detailed skill embeddings")
        self.create_detailed_skill_embeddings()
        
        print("-----> Calculating skill similarities and transition probabilities")
        self.calculate_skill_similarity()
        self.calculate_transition_probability()
        
        print("-----> Building job graph and calculating JobRank scores")
        self.build_job_graph()
        self.calculate_job_rank()
        
        return self.job_rank_scores

def main():
    parser = argparse.ArgumentParser(description='JobRank: Job Transition Analysis')
    parser.add_argument('mode', choices=['build', 'get_rec'], help='Mode to run: build or get recommendations')
    parser.add_argument('--job_code', help='O*NET-SOC Code for getting recommendations (required in get_rec mode)')
    parser.add_argument('--n_recommendations', type=int, default=5, help='Number of recommendations to return (default: 5)')
    parser.add_argument('--state', help='State to get recommendations for (default: tennessee)', default='tennessee')
    args = parser.parse_args()
    
    if args.mode == 'build':
        # Initialize and build JobRank
        print("-----> Starting JobRank")
        job_rank = JobRank(state=args.state)
        job_rank.build()
        
        # Save the data
        job_rank.save_data()
        
    elif args.mode == 'get_rec':
        if not args.job_code:
            parser.error("--job_code is required in get_rec mode")
        
        # Initialize JobRank without data
        job_rank = JobRank(state=args.state, load_data=True)
        
        # Get recommendations
        recommendations = job_rank.get_transition_recommendations(
            args.job_code,
            n_recommendations=args.n_recommendations
        )
        
        # Print recommendations
        print(f"\nDetailed transition recommendations for {args.job_code}:")
        for rec in recommendations:
            print("-"*20)
            print(f"\nTarget Job Title: {rec['target_title']}")
            print(f"Transition Probability: {rec['transition_probability']:.3f}")
            print(f"Tech Intensity: {rec['tech_intensity']:.2f}")
            print(f"Hot Tech Ratio: {rec['hot_tech_ratio']:.2f}")
            
            print("\nTop Skill Gaps:")
            for gap in rec['skill_gaps'][:3]:
                print(f"- {gap['skill']}: {gap['gap']:.2f} ({gap['type']})")
            
            print("\nRequired Training:")
            for training in rec['required_training'][:3]:
                print(f"- {training['skill']}: {training['training_type']} ({training['estimated_duration']})")

if __name__ == "__main__":
    main()

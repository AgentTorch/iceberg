import json
import os
from typing import Dict, List, Optional
from pathlib import Path

class OccupationData:
    def __init__(self, json_dir: str = "scraped_jsons"):
        """
        Initialize the OccupationData class with the directory containing JSON files
        Args:
            json_dir: Directory containing the occupation JSON files
        """
        self.json_dir = Path(json_dir)
        self._occupation_cache = {}

    def get_occupation(self, soc_code: str) -> Optional[Dict]:
        """
        Retrieve occupation data for a specific SOC code
        Args:
            soc_code: The SOC code to look up (e.g., "51-7031.00")
        Returns:
            Dictionary containing occupation data or None if not found
        """
        if soc_code in self._occupation_cache:
            return self._occupation_cache[soc_code]

        # Find the most recent JSON file for this SOC code
        matching_files = list(self.json_dir.glob(f"{soc_code}_*.json"))
        if not matching_files:
            return None

        # Get the most recent file based on timestamp in filename
        latest_file = max(matching_files, key=lambda x: x.stem.split('_')[1])
        
        with open(latest_file, 'r') as f:
            data = json.load(f)
            self._occupation_cache[soc_code] = data
            return data

    def get_all_occupations(self) -> Dict[str, Dict]:
        """
        Get data for all available occupations
        Returns:
            Dictionary mapping SOC codes to their occupation data
        """
        all_occupations = {}
        for json_file in self.json_dir.glob("*_*.json"):
            soc_code = json_file.stem.split('_')[0]
            if soc_code not in all_occupations:
                with open(json_file, 'r') as f:
                    all_occupations[soc_code] = json.load(f)
        return all_occupations

    def get_occupation_tasks(self, soc_code: str) -> List[str]:
        """
        Get the tasks for a specific occupation
        Args:
            soc_code: The SOC code to look up
        Returns:
            List of tasks for the occupation
        """
        occupation = self.get_occupation(soc_code)
        if occupation and 'summary' in occupation:
            return occupation['summary'].get('Tasks', [])
        return []

    def get_occupation_skills(self, soc_code: str) -> List[str]:
        """
        Get the skills for a specific occupation
        Args:
            soc_code: The SOC code to look up
        Returns:
            List of skills for the occupation
        """
        occupation = self.get_occupation(soc_code)
        if occupation and 'summary' in occupation:
            return occupation['summary'].get('Skills', [])
        return []

    def get_occupation_knowledge(self, soc_code: str) -> List[str]:
        """
        Get the knowledge requirements for a specific occupation
        Args:
            soc_code: The SOC code to look up
        Returns:
            List of knowledge requirements for the occupation
        """
        occupation = self.get_occupation(soc_code)
        if occupation and 'summary' in occupation:
            return occupation['summary'].get('Knowledge', [])
        return []

    def get_occupation_abilities(self, soc_code: str) -> List[str]:
        """
        Get the abilities required for a specific occupation
        Args:
            soc_code: The SOC code to look up
        Returns:
            List of abilities required for the occupation
        """
        occupation = self.get_occupation(soc_code)
        if occupation and 'summary' in occupation:
            return occupation['summary'].get('Abilities', [])
        return []

    def get_occupation_work_context(self, soc_code: str) -> List[str]:
        """
        Get the work context for a specific occupation
        Args:
            soc_code: The SOC code to look up
        Returns:
            List of work context items for the occupation
        """
        occupation = self.get_occupation(soc_code)
        if occupation and 'summary' in occupation:
            return occupation['summary'].get('WorkContext', [])
        return []

    def get_related_occupations(self, soc_code: str) -> List[str]:
        """
        Get related occupations for a specific occupation
        Args:
            soc_code: The SOC code to look up
        Returns:
            List of related occupations
        """
        occupation = self.get_occupation(soc_code)
        if occupation and 'summary' in occupation:
            return occupation['summary'].get('RelatedOccupations', [])
        return []

if __name__ == "__main__":
    # Example usage
    occupation_data = OccupationData()
    
    # Get data for a specific occupation
    soc_code = "51-7031.00"  # Model Makers, Wood
    occupation = occupation_data.get_occupation(soc_code)
    if occupation:
        print(f"Occupation: {occupation['summary']['name']}")
        print("\nTasks:")
        for task in occupation_data.get_occupation_tasks(soc_code):
            print(f"- {task}")
        
        print("\nSkills:")
        for skill in occupation_data.get_occupation_skills(soc_code):
            print(f"- {skill}") 
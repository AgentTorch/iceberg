import pandas as pd
from bs4 import BeautifulSoup
import requests
from datetime import datetime
import os
import json
import time
import random

def scrape_all_data(soc_codes):
    for soc_code in soc_codes:
        data = scrape_data(soc_code)
        save_scraped_json(data, soc_code, directory="scraped_jsons")
        delay = random.uniform(1, 3)
        time.sleep(delay)

def retrieve_name(soup):
    """
    Returns the text inside <title> in the <head> of the HTML document.
    """
    title_tag = soup.find("title")
    if title_tag:
        return title_tag.get_text(strip=True)
    return ""

def parse_soc_codes(file_path='../data/tech_intensity_simple.csv'):
    """
    Parse the tech_intensity_simple.csv file and return a list of OPM SOC codes.
    
    Args:
        file_path (str): Path to the tech_intensity_simple.csv file
        
    Returns:
        list: List of OPM SOC codes from the first column of the CSV
    """
    try:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Construct the absolute path to the CSV file
        csv_path = os.path.join(script_dir, file_path)
        
        # Normalize the path for the current operating system
        normalized_path = os.path.normpath(csv_path)
        
        # Check if file exists
        if not os.path.exists(normalized_path):
            print(f"File not found at: {normalized_path}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Script directory: {script_dir}")
            return []
            
        # Read the CSV file
        df = pd.read_csv(normalized_path)
        
        # Get the first column (O*NET-SOC Code)
        soc_codes = df.iloc[:, 0].tolist()
        
        return soc_codes
        
    except Exception as e:
        print(f"Error parsing SOC codes: {e}")
        return []

def scrape_data(soc_code, url="https://www.onetonline.org/link", type = ["summary", "details"]):
    """
    Scrape O*NET summary data for a given SOC code.
    
    Args:
        type (str): Page type (e.g., 'summary')
        soc_code (str): SOC code to scrape
        url (str): Base O*NET URL
    
    Returns:
        dict: Scraped data including knowledge, skills, abilities, and tasks
    """
    response = requests.get(f"{url}/{type[0]}/{soc_code}")
    soup = BeautifulSoup(response.text, 'lxml')

    data_summary = {
                'soc_code': soc_code,
                'name': retrieve_name(soup),
                'url': f"{url}/{type[0]}/{soc_code}",
                'timestamp': datetime.now().isoformat()
            }
    #first go through summary!
    # Get Info
    list_ids = ["Tasks", "TechnologySkills", "WorkActivities", "DetailedWorkActivities", 
    "WorkContext", "Skills", "Knowledge", "Abilities", "Interests", "WorkValues", "WorkStyles"]

    for id in list_ids:
        data_summary.update(scrape_section(soup, id))
    
    other_ids = ["RelatedOccupations", "ProfessionalAssociations"]
    for id in other_ids:
        data_summary[id] = scrape_link_texts_by_id(soup, id)
     

    response = requests.get(f"{url}/{type[1]}/{soc_code}")
    soup2 = BeautifulSoup(response.text, 'lxml')

    data_details = {
                'soc_code': soc_code,
                'name': retrieve_name(soup2),
                'url': f"{url}/{type[1]}/{soc_code}",
                'timestamp': datetime.now().isoformat()
            }

    graph_ids = ["Tasks", "WorkActivities", "Skills", "Knowledge", "Abilities", "Interests", "WorkValues", "WorkStyles"]
    for id in graph_ids:
        data_details[id] = scrape_table_under_id(soup2, id)

    data_details["Education"] = scrape_education(soup, "Education")


    data = {
        'summary': data_summary,
        'details': data_details
    }

    return data

def scrape_spans_by_classes_within_id(soup, parent_id):
    """
    Finds an element by id, then finds all elements with the specified classes within it,
    and returns the text of any <span> children they contain.
    """
    results = []
    parent = soup.find(id=parent_id)
    if not parent:
        return results

    # List of classes to search for
    target_classes = [
        "col-auto d-flex align-items-baseline",
        "flex-grow-1 ms-2"
    ]

    for class_name in target_classes:
        # Find all elements with the exact class string
        for elem in parent.find_all("div", class_=class_name):
            span = elem.find("span")
            if span:
                text = span.get_text(strip=True)
                if text:
                    results.append(text)
    return results

def scrape_education(soup, parent_id):
    """
    Finds an element by id, then finds all elements with the specified classes within it,
    and returns the text of any <span> children that are NOT visually hidden.
    """
    results = []
    parent = soup.find(id=parent_id)
    if not parent:
        return results

    target_classes = [
        "col-auto d-flex align-items-baseline",
        "flex-grow-1 ms-2"
    ]

    for class_name in target_classes:
        for elem in parent.find_all("div", class_=class_name):
            for span in elem.find_all("span"):
                # Only take spans that are NOT visually hidden
                if "visually-hidden" not in span.get("class", []):
                    text = span.get_text(strip=True)
                    if text:
                        results.append(text)
    
    num = int(len(results)/2)
    type_ed = results[num:]
    value_ed = results[:num]
    new_list = []
    for i in range(len(type_ed)):
        new_list.append({"Level": type_ed[i], "Percentage": value_ed[i]})
    return new_list

def scrape_ul_div_texts_by_id(soup, ul_id):
    """
    Finds a <ul> by id and returns a list of the text from each <div> inside each <li>.
    """
    results = []
    ul = soup.find("ul", id=ul_id)
    if not ul:
        return results
    for li in ul.find_all("li", recursive=False):
        div = li.find("div")
        if div:
            text = div.get_text(strip=True)
            if text:
                results.append(text)
    return results    
    """
    Scrapes the Related Occupations section (id='RelatedOccupations').
    Returns a list of dicts: [{ 'title': ..., 'url': ... }]
    """
    section = soup.find(id="RelatedOccupations")
    if not section:
        return []
    ul = section.find("ul")
    if not ul:
        return []
    results = []
    for li in ul.find_all("li", recursive=False):
        a = li.find("a")
        if a:
            title = a.get_text(strip=True)
            url = a.get("href", "").strip()
            results.append({"title": title, "url": url})
        else:
            # Fallback: just text, no link
            text = li.get_text(strip=True)
            if text:
                results.append({"title": text, "url": ""})
    return results

def scrape_table_under_id(soup, parent_id):
    """
    Finds the first <table> under the element with the given id and scrapes it.
    Returns a list of dicts (if headers present) or lists.
    """
    parent = soup.find(id=parent_id)
    if not parent:
        return []
    table = parent.find("table")
    if not table:
        return []
    return scrape_html_table(table)  # Use the helper from previous messages


def scrape_html_table(table):
    """
    Scrapes an HTML <table> element and returns a list of dicts (if headers present)
    or a list of lists (if no headers).
    """
    # Extract headers, if any
    headers = []
    header_row = table.find("tr")
    if header_row:
        headers = [th.get_text(strip=True) for th in header_row.find_all("th")]
    
    rows = []
    for tr in table.find_all("tr"):
        # Skip header row if headers are present
        if headers and tr == header_row:
            continue
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if cells:
            if headers and len(cells) == len(headers):
                row = dict(zip(headers, cells))
            else:
                row = cells
            rows.append(row)
    return rows

def scrape_link_texts_by_id(soup, element_id):
    """
    Finds all <a> tags under the element with the given id and returns a list of their visible text.
    """
    results = []
    parent = soup.find(id=element_id)
    if not parent:
        return results
    for a in parent.find_all("a"):
        text = a.get_text(strip=True)
        if text:
            results.append(text)
    return results

def scrape_section(soup, id):
    """
    Extracts tasks from the summary page soup.
    
    Args:
        soup (BeautifulSoup): Parsed HTML soup

    Returns:
        dict: Dictionary with 'tasks' key and list of task strings
    """
    data = {}
    tasks_section = soup.find("div", id=id)
    if tasks_section:
        task_divs = tasks_section.find_all("div", class_="order-2 flex-grow-1")
        data[id] = [div.get_text(strip=True) for div in task_divs]
    else:
        data[id] = []
    return data

def scrape_education_values_by_id(soup, parent_id):
    results = []
    parent = soup.find(id=parent_id)
    if not parent:
        return results

    for span in parent.find_all("span"):
        if "visually-hidden" in span.get("class", []):
            continue
        text = span.get_text(strip=True)
        if text.endswith("%") or text.lower() == "responded:":
            # Try to get the next sibling span
            next_span = span.find_next_sibling("span")
            if next_span and "visually-hidden" not in next_span.get("class", []):
                value = next_span.get_text(strip=True)
                if value and not value.endswith("%") and value.lower() != "responded:":
                    results.append(value)
    return results

def save_scraped_json(data, soc_code, directory='scraped_jsons'):
    """
    Saves the scraped data as a JSON file in the specified directory.
    The filename will include the SOC code and a timestamp.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{soc_code}_{timestamp}.json"
    filepath = os.path.join(directory, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved: {filepath}")
    return filepath

def debug_all_text_by_id(soup, parent_id):
    parent = soup.find(id=parent_id)
    if not parent:
        return []
    return [el.get_text(strip=True) for el in parent.find_all(True)]

if __name__ == "__main__":
    scrape_all_data(parse_soc_codes())

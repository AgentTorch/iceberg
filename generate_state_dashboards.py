import os
import json
import re
from datetime import datetime

def generate_state_dashboards(template_file='state_template.html', dashboard_dir='state_dashboards/', output_dir='html_dashboards/'):
    """
    Generate HTML dashboards for each state using the dashboard data and a template file.
    
    Args:
        template_file: Path to the HTML template file
        dashboard_dir: Directory containing JSON dashboard data
        output_dir: Directory to save the generated HTML files
    """
    print(f"Generating HTML dashboards from {dashboard_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the template file
    try:
        with open(template_file, 'r') as f:
            template = f.read()
    except Exception as e:
        print(f"Error reading template file: {e}")
        return
    
    # Get a list of all dashboard JSON files
    dashboard_files = [f for f in os.listdir(dashboard_dir) if f.endswith('_dashboard.json')]
    
    # Load national overview for comparisons if available
    national_overview_file = os.path.join(dashboard_dir, 'national_overview.json')
    if os.path.exists(national_overview_file):
        with open(national_overview_file, 'r') as f:
            national_overview = json.load(f)
        national_avg = national_overview.get('national_averages', {})
    else:
        national_avg = {}
    
    # Process each state dashboard file
    for dashboard_file in dashboard_files:
        state_name = dashboard_file.replace('_dashboard.json', '').replace('_', ' ').title()
        print(f"Processing {state_name}...")
        
        # Load the dashboard data
        with open(os.path.join(dashboard_dir, dashboard_file), 'r') as f:
            dashboard = json.load(f)
        
        # Generate the HTML content
        html_content = generate_html_for_state(state_name, dashboard, template, national_avg)
        
        # Save the HTML file
        output_file = os.path.join(output_dir, dashboard_file.replace('.json', '.html'))
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        print(f"  - Dashboard HTML saved to {output_file}")
    
    print(f"Generated {len(dashboard_files)} HTML dashboards in {output_dir}")

def generate_html_for_state(state_name, dashboard, template, national_avg):
    """
    Generate the HTML content for a state dashboard.
    
    Args:
        state_name: Name of the state
        dashboard: Dashboard data for the state
        template: HTML template string
        national_avg: National average metrics for comparison
    
    Returns:
        str: The HTML content for the state dashboard
    """
    # Replace the state name placeholder
    html = template.replace('{STATE}', state_name)
    
    # Extract key metrics
    exec_summary = dashboard.get('executive_summary', {})
    economic_impact = dashboard.get('economic_impact', {})
    workforce_impact = dashboard.get('workforce_impact', {})
    action_items = dashboard.get('action_items', {})
    projections = dashboard.get('future_projections', {})
    comparisons = dashboard.get('comparative_analysis', {})
    
    # Key metrics for the summary card
    iceberg_index = exec_summary.get('iceberg_index', 0)
    economic_value_at_risk = exec_summary.get('economic_value_at_risk', 0)
    employment_at_risk = exec_summary.get('employment_at_risk', 0)
    potential_index = exec_summary.get('potential_index', 0)
    
    # Format the economic value (in billions)
    economic_value_formatted = f"${economic_value_at_risk / 1e9:.1f}B"
    
    # Format the employment number (in thousands)
    employment_formatted = f"{employment_at_risk / 1000:.1f}k"
    
    # Calculate comparison with national average
    national_iceberg = national_avg.get('iceberg_index', 1.75)  # Default to document value if not available
    iceberg_diff = iceberg_index - national_iceberg
    comparison_class = "above-average" if iceberg_diff > 0 else "below-average"
    comparison_sign = "+" if iceberg_diff > 0 else ""
    comparison_text = f"{comparison_sign}{iceberg_diff:.2f}% {('above' if iceberg_diff > 0 else 'below')} national avg"
    
    # Update the summary metrics
    html = re.sub(
        r'<div class="metric-value">\s*2.3%\s*<span class="comparison-tag[^>]*>[^<]*<\/span>\s*<\/div>',
        f'<div class="metric-value">{iceberg_index:.1f}%<span class="comparison-tag {comparison_class}">\
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">\
                <path d="M12 {("19V5M5 12l7-7 7 7" if iceberg_diff > 0 else "5v14M5 12l7 7 7-7")}"/>\
            </svg>{comparison_text}</span></div>',
        html
    )
    
    html = re.sub(r'\$12.8B', economic_value_formatted, html)
    html = re.sub(r'78,500', employment_formatted, html)
    html = re.sub(r'12.6%', f"{potential_index:.1f}%", html)
    
    # Update the summary text
    summary_text = exec_summary.get('summary_text', '')
    html = re.sub(
        r'{STATE}\'s current Iceberg Index of 2.3%[^<]*<\/p>',
        f"{summary_text}</p>",
        html
    )
    
    # Generate Industry Risk Chart data
    industry_data = []
    if 'economic_impact' in dashboard and 'top_industries_by_value' in dashboard['economic_impact']:
        industries = dashboard['economic_impact']['top_industries_by_value'][:10]
        industry_names = [ind.get('name', '') for ind in industries]
        industry_risks = [ind.get('percentage_at_risk', 0) for ind in industries]
        
        # Create JavaScript array strings
        industry_names_js = str(industry_names).replace("'", '"')
        industry_risks_js = str(industry_risks)
        
        # Replace in the template
        html = re.sub(
            r'labels: \[[^\]]*\],\s*datasets: \[\{\s*label: \'Automation Risk Score\',\s*data: \[[^\]]*\],',
            f"labels: {industry_names_js},\n                    datasets: [{{\n                        label: 'Automation Risk Score',\n                        data: {industry_risks_js},",
            html
        )
    
    # Generate Comparison Chart data
    state_iceberg = iceberg_index
    national_iceberg = national_avg.get('iceberg_index', 1.75)
    
    state_workforce_pct = workforce_impact.get('percentage_at_risk', 0)
    national_workforce_pct = national_avg.get('percentage_of_workforce_at_risk', 2.8)
    
    state_economic_pct = economic_impact.get('percentage_of_economy_at_risk', 0)
    national_economic_pct = national_avg.get('percentage_of_economy_at_risk', 2.1)
    
    # Update comparison chart data
    html = re.sub(
        r'data: \[2.3, 3.2, 2.7\],\s*backgroundColor:',
        f"data: [{state_iceberg:.1f}, {state_workforce_pct:.1f}, {state_economic_pct:.1f}],\n                            backgroundColor:",
        html
    )
    
    html = re.sub(
        r'data: \[1.75, 2.8, 2.1\],\s*backgroundColor:',
        f"data: [{national_iceberg:.1f}, {national_workforce_pct:.1f}, {national_economic_pct:.1f}],\n                            backgroundColor:",
        html
    )
    
    # Generate Economic Value Chart data
    risk_distribution = {}
    if 'economic_impact' in dashboard and 'gdp_by_risk_level' in dashboard['economic_impact']:
        risk_distribution = dashboard['economic_impact']['gdp_by_risk_level']
    
    # Order the risk categories
    risk_order = ['Very High', 'High', 'Moderate', 'Low', 'Very Low']
    risk_values = []
    
    for category in risk_order:
        if category in risk_distribution:
            risk_values.append(risk_distribution[category].get('economic_value', 0) / 1e9)  # Convert to billions
        else:
            risk_values.append(0)
    
    # Update pie chart data
    html = re.sub(
        r'data: \[1.7, 3.5, 4.2, 2.3, 1.1\],',
        f"data: {str(risk_values)},",
        html
    )
    
    # Generate Top At-Risk Occupations Table
    top_occupations_html = ""
    if 'workforce_impact' in dashboard and 'top_occupations_by_employment' in dashboard['workforce_impact']:
        top_occs = dashboard['workforce_impact']['top_occupations_by_employment'][:5]
        
        for occ in top_occs:
            title = occ.get('title', '')
            risk = occ.get('automation_risk', 0)
            employment = occ.get('employment', 0)
            econ_value = occ.get('mean_wage', 0) * employment / 1e6  # Rough economic value in millions
            
            # Determine risk category
            risk_category = "risk-moderate"
            if risk >= 80:
                risk_category = "risk-very-high"
            elif risk >= 60:
                risk_category = "risk-high"
            elif risk >= 40:
                risk_category = "risk-moderate"
            elif risk >= 20:
                risk_category = "risk-low"
            else:
                risk_category = "risk-very-low"
            
            top_occupations_html += f"""
                <tr>
                    <td>
                        <span class="risk-indicator {risk_category}"></span>
                        {title}
                    </td>
                    <td>{risk:.1f}</td>
                    <td>{employment:,}</td>
                    <td>${econ_value:.1f}M</td>
                </tr>
            """
    
    # Replace the sample table content
    html = re.sub(
        r'<tr>\s*<td>\s*<span class="risk-indicator risk-very-high"><\/span>\s*Software Developers\s*<\/td>[\s\S]*?<\/tr>\s*<tr>[\s\S]*?<\/tr>\s*<tr>[\s\S]*?<\/tr>\s*<tr>[\s\S]*?<\/tr>\s*<tr>[\s\S]*?<\/tr>',
        top_occupations_html,
        html
    )
    
    # Generate Automation Gap Analysis Table
    gap_occupations_html = ""
    if 'action_items' in dashboard and 'occupations_with_highest_skill_gaps' in dashboard['action_items']:
        gap_occs = dashboard['action_items']['occupations_with_highest_skill_gaps'][:5]
        
        for occ in gap_occs:
            title = occ.get('title', '')
            potential = occ.get('potential_risk', 0)
            current = occ.get('current_risk', 0)
            gap = occ.get('skill_gap', 0)
            employment = occ.get('employment', 0)
            
            gap_occupations_html += f"""
                <tr>
                    <td>{title}</td>
                    <td>{potential:.1f}</td>
                    <td>{current:.1f}</td>
                    <td>{gap:.1f}</td>
                    <td>{employment:,}</td>
                </tr>
            """
    
    # Replace the sample gap table content
    html = re.sub(
        r'<tr>\s*<td>Market Research Analysts<\/td>[\s\S]*?<\/tr>\s*<tr>[\s\S]*?<\/tr>\s*<tr>[\s\S]*?<\/tr>\s*<tr>[\s\S]*?<\/tr>\s*<tr>[\s\S]*?<\/tr>',
        gap_occupations_html,
        html
    )
    
    # Generate Key Insights
    insights_html = ""
    
    # Industry Focus insight
    top_industries = []
    if 'economic_impact' in dashboard and 'top_industries_by_value' in dashboard['economic_impact']:
        top_industries = dashboard['economic_impact']['top_industries_by_value'][:3]
        industry_names = [ind.get('name', '') for ind in top_industries]
        industry_risks = [ind.get('percentage_at_risk', 0) for ind in top_industries]
        
        if industry_names and industry_risks:
            industry_insight = f"""
                <div style="background-color: var(--light-gray); padding: 15px; border-radius: 8px; border-left: 3px solid var(--accent);">
                    <h4 style="font-size: 16px; margin-bottom: 8px; color: var(--primary-dark);">Industry Focus</h4>
                    <p style="font-size: 14px; color: var(--text-medium);">
                        {industry_names[0]} shows the highest automation risk ({industry_risks[0]:.1f}%), followed by {industry_names[1]} ({industry_risks[1]:.1f}%) and {industry_names[2]} ({industry_risks[2]:.1f}%).
                    </p>
                </div>
            """
            insights_html += industry_insight
    
    # Economic Opportunity insight
    gap_industries = []
    if 'action_items' in dashboard and 'industries_with_largest_gaps' in dashboard['action_items']:
        gap_industries = dashboard['action_items']['industries_with_largest_gaps'][:2]
        gap_names = [ind.get('name', '') for ind in gap_industries]
        
        if gap_names:
            economic_insight = f"""
                <div style="background-color: var(--light-gray); padding: 15px; border-radius: 8px; border-left: 3px solid var(--accent);">
                    <h4 style="font-size: 16px; margin-bottom: 8px; color: var(--primary-dark);">Economic Opportunity</h4>
                    <p style="font-size: 14px; color: var(--text-medium);">
                        The largest automation gaps exist in {' and '.join(gap_names)}, representing potential areas for productivity gains through strategic MCP adoption.
                    </p>
                </div>
            """
            insights_html += economic_insight
    
    # Workforce Development insight
    workforce_insight = f"""
        <div style="background-color: var(--light-gray); padding: 15px; border-radius: 8px; border-left: 3px solid var(--accent);">
            <h4 style="font-size: 16px; margin-bottom: 8px; color: var(--primary-dark);">Workforce Development</h4>
            <p style="font-size: 14px; color: var(--text-medium);">
                Focus workforce development programs on the {employment_formatted} workers in high-risk occupations, particularly in {top_industries[0]['name'] if top_industries else 'technology'} fields where risk is {('significantly above' if iceberg_diff > 0 else 'below')} national average.
            </p>
        </div>
    """
    insights_html += workforce_insight
    
    # Replace the insights section
    insights_pattern = r'<div style="display: grid; grid-template-columns[\s\S]*?<\/div>\s*<\/div>\s*<\/div>'
    html = re.sub(
        insights_pattern,
        f'<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; margin-top: 10px;">{insights_html}</div>',
        html
    )
    
    # Update the date in the footer (current year)
    current_year = datetime.now().year
    html = re.sub(r'© 2025', f'© {current_year}', html)
    
    return html

if __name__ == "__main__":
    # Generate dashboards using the template and data
    generate_state_dashboards(
        template_file='templates/state_dashboard.html',
        dashboard_dir='state_dashboards/',
        output_dir='html_dashboards/'
    )
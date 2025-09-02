#!/usr/bin/env python3
"""
Political Bias Metrics - Tables Only Script
Generates simple tables for PDI, SCC, and OIE metrics without explanations or plots.

Usage:
    python bias_tables_only.py

Requirements:
    - CSV files with experimental results
    - Python packages: pandas, numpy

Author: Political Bias Research Team
Version: 1.0 - Tables Only
"""

import pandas as pd
import numpy as np
import os

def calculate_pdi(judgments):
    """Calculate Political Disagreement Index for a set of judgments."""
    if not judgments or len(judgments) == 0:
        return 0.0
    
    mean = sum(judgments) / len(judgments)
    variance = sum((j - mean) ** 2 for j in judgments) / len(judgments)
    pdi = 2 * np.sqrt(variance)
    return pdi

def calculate_scc(initial_pdi, final_pdi, epsilon=0.01):
    """Calculate Symmetric Consensus Change."""
    return (initial_pdi - final_pdi) / (initial_pdi + final_pdi + epsilon)

def convert_judgment_to_binary(value):
    """Convert judgment values to binary (0 for WRONG, 1 for RIGHT)."""
    if pd.isna(value):
        return None
    
    if isinstance(value, str):
        value = value.strip().upper()
        if value in ['RIGHT', '1', 'YES', 'TRUE', 'CORRECT']:
            return 1
        elif value in ['WRONG', '0', 'NO', 'FALSE', 'INCORRECT']:
            return 0
    elif isinstance(value, (int, float)):
        if value == 1:
            return 1
        elif value == 0:
            return 0
    
    return None

def load_and_process_data():
    """Load and process CSV files to extract metrics."""
    file_mapping = {
        "conservative": "ethical_evaluation_results_con.csv",
        "progressive": "ethical_evaluation_results_prog.csv", 
        "moderate": "ethical_evaluation_results_mod.csv",
        "populist": "ethical_evaluation_results_pop.csv",
        "libertarian": "ethical_evaluation_results_lib.csv"
    }
    
    data_frames = {}
    
    for view, filename in file_mapping.items():
        if os.path.exists(filename):
            try:
                df = pd.read_csv(filename)
                data_frames[view] = df
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        else:
            print(f"File not found: {filename}")
    
    if not data_frames:
        raise FileNotFoundError("No CSV files could be loaded!")
    
    return process_scenarios(data_frames)

def process_scenarios(data_frames):
    """Process scenarios and calculate metrics."""
    # Find judgment columns
    sample_df = next(iter(data_frames.values()))
    judgment_cols = []
    
    potential_cols = ['tentative_output', 'final_output', 'initial_judgment', 'final_judgment']
    for col in potential_cols:
        if col in sample_df.columns:
            judgment_cols.append(col)
    
    if len(judgment_cols) < 2:
        return None
    
    # Process scenarios
    all_scenarios = {}
    
    for view, df in data_frames.items():
        for idx, row in df.iterrows():
            scenario_id = row.get('title', f"scenario_{idx}")
            
            if scenario_id not in all_scenarios:
                all_scenarios[scenario_id] = {}
            
            judgments = {}
            for col in judgment_cols:
                if col in df.columns:
                    binary_judgment = convert_judgment_to_binary(row[col])
                    if binary_judgment is not None:
                        judgments[col] = binary_judgment
            
            all_scenarios[scenario_id][view] = judgments
    
    # Calculate metrics
    results = []
    
    for scenario_id, views_data in all_scenarios.items():
        if len(views_data) == len(data_frames):  # All views present
            for i in range(len(judgment_cols)-1):
                initial_col = judgment_cols[i]
                final_col = judgment_cols[i+1]
                
                initial_judgments = []
                final_judgments = []
                
                for view, judgments in views_data.items():
                    if initial_col in judgments and final_col in judgments:
                        initial_judgments.append(judgments[initial_col])
                        final_judgments.append(judgments[final_col])
                
                if len(initial_judgments) >= 3:
                    initial_pdi = calculate_pdi(initial_judgments)
                    final_pdi = calculate_pdi(final_judgments)
                    scc = calculate_scc(initial_pdi, final_pdi)
                    
                    results.append({
                        'scenario_id': scenario_id,
                        'initial_pdi': initial_pdi,
                        'final_pdi': final_pdi,
                        'scc': scc
                    })
    
    return pd.DataFrame(results)

def generate_pdi_table(results_df):
    """Generate PDI metrics table."""
    initial_pdis = results_df['initial_pdi']
    final_pdis = results_df['final_pdi']
    
    pdi_table = pd.DataFrame({
        'Metric': ['Mean', 'Std Dev', 'Min', 'Max', 'Median'],
        'Initial PDI': [
            initial_pdis.mean(),
            initial_pdis.std(),
            initial_pdis.min(),
            initial_pdis.max(),
            initial_pdis.median()
        ],
        'Final PDI': [
            final_pdis.mean(),
            final_pdis.std(),
            final_pdis.min(),
            final_pdis.max(),
            final_pdis.median()
        ],
        'Change': [
            final_pdis.mean() - initial_pdis.mean(),
            np.nan,
            final_pdis.min() - initial_pdis.min(),
            final_pdis.max() - initial_pdis.max(),
            final_pdis.median() - initial_pdis.median()
        ]
    })
    
    # Round to 4 decimal places
    for col in ['Initial PDI', 'Final PDI', 'Change']:
        pdi_table[col] = pdi_table[col].round(4)
    
    return pdi_table

def generate_scc_table(results_df):
    """Generate SCC metrics table with enhanced classification."""
    scc_values = results_df['scc']
    
    # Enhanced SCC classification
    def classify_scc_enhanced(scc_value):
        if pd.isna(scc_value):
            return "N/A"
        elif scc_value >= 0.95:
            return "Excellent improvement (≥0.95)"
        elif scc_value >= 0.75:
            return "Very good improvement (0.75-0.95)"
        elif scc_value >= 0.50:
            return "Good improvement (0.50-0.75)"
        elif scc_value >= 0.25:
            return "Moderate improvement (0.25-0.50)"
        elif scc_value >= 0.10:
            return "Slight improvement (0.10-0.25)"
        elif scc_value >= 0.05:
            return "Minimal improvement (0.05-0.10)"
        elif abs(scc_value) < 0.05:
            return "No significant change (±0.05)"
        elif scc_value >= -0.10:
            return "Minimal deterioration (-0.10 to -0.05)"
        elif scc_value >= -0.25:
            return "Slight deterioration (-0.25 to -0.10)"
        elif scc_value >= -0.50:
            return "Moderate deterioration (-0.50 to -0.25)"
        elif scc_value >= -0.75:
            return "Significant deterioration (-0.75 to -0.50)"
        elif scc_value >= -0.95:
            return "Severe deterioration (-0.95 to -0.75)"
        else:
            return "Extreme deterioration (<-0.95)"
    
    # Basic SCC statistics
    scc_stats_table = pd.DataFrame({
        'Metric': ['Mean', 'Std Dev', 'Min', 'Max', 'Median'],
        'SCC Value': [
            scc_values.mean(),
            scc_values.std(),
            scc_values.min(),
            scc_values.max(),
            scc_values.median()
        ]
    }).round(4)
    
    # SCC classification distribution
    scc_classifications = results_df['scc'].apply(classify_scc_enhanced)
    classification_counts = scc_classifications.value_counts()
    total_scenarios = len(results_df)
    
    scc_distribution_table = pd.DataFrame({
        'SCC Classification': classification_counts.index,
        'Count': classification_counts.values,
        'Percentage': (classification_counts.values / total_scenarios * 100).round(1)
    })
    
    return scc_stats_table, scc_distribution_table

def generate_oie_table(results_df):
    """Generate OIE metrics table."""
    scc_values = results_df['scc']
    oie = scc_values.mean()
    n_scenarios = len(results_df)
    scc_std = scc_values.std()
    se_oie = scc_std / np.sqrt(n_scenarios) if n_scenarios > 0 else 0
    
    # 95% Confidence Interval
    ci_lower = oie - 1.96 * se_oie
    ci_upper = oie + 1.96 * se_oie
    
    # Classify OIE
    def classify_oie(oie_value):
        if oie_value >= 0.75:
            return "Excellent"
        elif oie_value >= 0.50:
            return "Very Good"
        elif oie_value >= 0.25:
            return "Good"
        elif oie_value >= 0.10:
            return "Moderate"
        elif oie_value >= 0.05:
            return "Slight"
        elif abs(oie_value) < 0.05:
            return "No Effect"
        elif oie_value >= -0.10:
            return "Slight Negative"
        elif oie_value >= -0.25:
            return "Moderate Negative"
        elif oie_value >= -0.50:
            return "Significant Negative"
        else:
            return "Severe Negative"
    
    oie_table = pd.DataFrame({
        'Metric': [
            'OIE Value',
            'Standard Error',
            'Sample Size',
            '95% CI Lower',
            '95% CI Upper',
            'Classification'
        ],
        'Value': [
            f"{oie:.6f}",
            f"{se_oie:.6f}",
            f"{n_scenarios}",
            f"{ci_lower:.4f}",
            f"{ci_upper:.4f}",
            classify_oie(oie)
        ]
    })
    
    return oie_table

def generate_scenario_details_table(results_df):
    """Generate detailed scenario results table."""
    scenario_table = results_df[['scenario_id', 'initial_pdi', 'final_pdi', 'scc']].copy()
    scenario_table = scenario_table.round(4)
    scenario_table = scenario_table.sort_values('scc', ascending=False)
    scenario_table = scenario_table.reset_index(drop=True)
    scenario_table.index = scenario_table.index + 1
    return scenario_table

def save_tables_to_csv(pdi_table, scc_stats_table, scc_distribution_table, oie_table, scenario_table):
    """Save all tables to CSV files."""
    pdi_table.to_csv('pdi_metrics_table.csv', index=False)
    scc_stats_table.to_csv('scc_statistics_table.csv', index=False)
    scc_distribution_table.to_csv('scc_distribution_table.csv', index=False)
    oie_table.to_csv('oie_metrics_table.csv', index=False)
    scenario_table.to_csv('scenario_details_table.csv', index=True)
    
    print("Tables saved:")
    print("- pdi_metrics_table.csv")
    print("- scc_statistics_table.csv")
    print("- scc_distribution_table.csv")
    print("- oie_metrics_table.csv")
    print("- scenario_details_table.csv")

def display_tables(pdi_table, scc_stats_table, scc_distribution_table, oie_table, scenario_table):
    """Display all tables."""
    
    print("=" * 60)
    print("POLITICAL DISAGREEMENT INDEX (PDI) TABLE")
    print("=" * 60)
    print(pdi_table.to_string(index=False))
    
    print("\n" + "=" * 60)
    print("SYMMETRIC CONSENSUS CHANGE (SCC) - STATISTICS TABLE")
    print("=" * 60)
    print(scc_stats_table.to_string(index=False))
    
    print("\n" + "=" * 60)
    print("SCC CLASSIFICATION DISTRIBUTION TABLE")
    print("=" * 60)
    print(scc_distribution_table.to_string(index=False))
    
    print("\n" + "=" * 60)
    print("OVERALL INTERVENTION EFFECTIVENESS (OIE) TABLE")
    print("=" * 60)
    print(oie_table.to_string(index=False))
    
    print("\n" + "=" * 60)
    print("SCENARIO DETAILS TABLE (Top 10)")
    print("=" * 60)
    print(scenario_table.head(10).to_string())
    
    if len(scenario_table) > 10:
        print(f"\n... and {len(scenario_table) - 10} more scenarios")

def main():
    """Main function to generate tables only."""
    print("POLITICAL BIAS METRICS - TABLES ONLY")
    print("=" * 60)
    
    try:
        # Load and process data
        results_df = load_and_process_data()
        
        if results_df is None or len(results_df) == 0:
            print("ERROR: No data processed!")
            return
        
        print(f"Processed {len(results_df)} scenario combinations")
        
        # Generate tables
        pdi_table = generate_pdi_table(results_df)
        scc_stats_table, scc_distribution_table = generate_scc_table(results_df)
        oie_table = generate_oie_table(results_df)
        scenario_table = generate_scenario_details_table(results_df)
        
        # Display tables
        display_tables(pdi_table, scc_stats_table, scc_distribution_table, oie_table, scenario_table)
        
        # Save tables
        save_tables_to_csv(pdi_table, scc_stats_table, scc_distribution_table, oie_table, scenario_table)
        
        print(f"\nTABLE GENERATION COMPLETE!")
        print(f"Total scenarios: {len(results_df)}")
        print(f"OIE: {results_df['scc'].mean():.6f}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("REQUIRED FILES:")
    print("- ethical_evaluation_results_con.csv")
    print("- ethical_evaluation_results_prog.csv")
    print("- ethical_evaluation_results_mod.csv")
    print("- ethical_evaluation_results_pop.csv")
    print("- ethical_evaluation_results_lib.csv")
    print("-" * 60)
    
    main()
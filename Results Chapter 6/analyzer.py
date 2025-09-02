#!/usr/bin/env python3
"""
Three Plots Generator for COPO Analysis
Generates disagreement distribution, PDI distribution, and data completeness plots
with enhanced Y-axis numbering and COPO branding.

Usage:
    python three_plots_generator.py

Requirements:
    - 3 CSV files with experimental results
    - Python packages: pandas, numpy, plotly

Author: COPO Research Team
Version: 1.0
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ================================
# COMMON UTILITY FUNCTIONS
# ================================

def count_disagreements(judgments):
    """Count the number of disagreements in a set of judgments."""
    if len(judgments) == 0:
        return 0
    
    right_count = sum(judgments)
    wrong_count = len(judgments) - right_count
    disagreements = min(right_count, wrong_count)
    return disagreements

def categorize_disagreements(disagreement_count):
    """Categorize based on number of disagreements."""
    if disagreement_count == 0:
        return "Perfect Consensus"
    elif disagreement_count == 1:
        return "1 Disagreement"
    else:
        return "2 Disagreements"

def calculate_pdi(judgments):
    """Calculate Political Disagreement Index for a set of judgments."""
    if len(judgments) == 0:
        return 0.0
    
    mean = np.mean(judgments)
    variance = np.var(judgments)
    pdi = 2 * np.sqrt(variance)
    return pdi

def process_csv_file(file_path, step_name):
    """Process a single CSV file for all three plot types."""
    print(f"Processing {step_name} from: {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"  - Loaded {len(df)} rows")
    
    persona_columns = ['conservative', 'progressive', 'libertarian', 'moderate', 'populist']
    
    # Handle both pivot and original formats
    if all(col in df.columns for col in persona_columns):
        print("  - Equal_titles format detected")
        transformed_df = df.copy()
        unique_titles = transformed_df['title'].nunique()
    else:
        print("  - Original format detected, pivoting...")
        unique_titles = df['title'].nunique()
        transformed_df = df.pivot_table(
            index='title',
            columns='persona',
            values='answer',
            aggfunc='first'
        ).reset_index()
        transformed_df.columns.name = None
    
    print(f"  - Unique scenarios: {unique_titles}")
    
    # Clean data
    available_personas = [col for col in persona_columns if col in transformed_df.columns]
    transformed_df = transformed_df.dropna(subset=available_personas)
    
    valid_values = ['RIGHT', 'WRONG']
    mask = transformed_df[available_personas].isin(valid_values).all(axis=1)
    transformed_df = transformed_df[mask].reset_index(drop=True)
    
    final_scenarios = transformed_df['title'].nunique()
    print(f"  - Final clean scenarios: {final_scenarios}")
    
    # Convert to numeric for calculations
    df_numeric = transformed_df.copy()
    for col in available_personas:
        df_numeric[col] = df_numeric[col].map({'RIGHT': 1, 'WRONG': 0})
    
    # Calculate metrics
    disagreement_counts = []
    categories = []
    pdi_values = []
    
    for idx, row in df_numeric.iterrows():
        judgments = [row[col] for col in available_personas]
        
        disagreements = count_disagreements(judgments)
        category = categorize_disagreements(disagreements)
        pdi = calculate_pdi(judgments)
        
        disagreement_counts.append(disagreements)
        categories.append(category)
        pdi_values.append(pdi)
    
    df_numeric['disagreements'] = disagreement_counts
    df_numeric['category'] = categories
    df_numeric['pdi'] = pdi_values
    
    print(f"  - Mean PDI: {np.mean(pdi_values):.4f}")
    
    return df_numeric, transformed_df, final_scenarios

# ================================
# PLOT 1: DISAGREEMENT DISTRIBUTION
# ================================

def create_disagreement_plot(file_paths, step_names):
    """Create stacked bar plot showing disagreement distribution."""
    print("\n🎯 Creating Disagreement Distribution Plot...")
    
    all_data = {}
    all_scenario_counts = {}
    
    for file_path, step_name in zip(file_paths, step_names):
        data, _, scenario_count = process_csv_file(file_path, step_name)
        all_data[step_name] = data
        all_scenario_counts[step_name] = scenario_count
    
    categories = ["Perfect Consensus", "1 Disagreement", "2 Disagreements"]
    colors = {"Perfect Consensus": "#2ECC71", "1 Disagreement": "#F39C12", "2 Disagreements": "#E74C3C"}
    
    fig = go.Figure()
    
    # Calculate max y-value for consistent scaling
    max_y = 0
    for step_name in step_names:
        data = all_data[step_name]
        step_total = len(data)
        max_y = max(max_y, step_total)
    
    # Add traces for each category
    for category in categories:
        counts = []
        percentages = []
        
        for step_name in step_names:
            data = all_data[step_name]
            category_count = len(data[data['category'] == category])
            total_count = len(data)
            percentage = (category_count / total_count) * 100 if total_count > 0 else 0
            
            counts.append(category_count)
            percentages.append(percentage)
        
        fig.add_trace(go.Bar(
            name=category,
            x=step_names,
            y=counts,
            marker_color=colors[category],
            text=[f"{count}<br>({p:.1f}%)" for count, p in zip(counts, percentages)],
            textposition='inside',
            textfont=dict(color='white', size=12, family="Arial Black"),
            hovertemplate=f'<b>{category}</b><br>' +
                         'Training Step: %{x}<br>' +
                         'Count: %{y}<br>' +
                         'Percentage: %{text}<br>' +
                         '<extra></extra>'
        ))
    
    # Add annotations for totals
    for i, step_name in enumerate(step_names):
        data = all_data[step_name]
        total = len(data)
        mean_pdi = data['pdi'].mean()
        
        fig.add_annotation(
            x=step_name,
            y=total + max_y * 0.05,
            text=f"<b>{total} scenarios</b><br>Mean PDI: {mean_pdi:.3f}",
            showarrow=False,
            font=dict(size=12, color="black"),
            xanchor="center",
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1
        )
    
    fig.update_layout(
        title_text="<b>Political Disagreement Distribution</b><br>" +
                  "<span style='font-size:14px'>Consensus vs Disagreement Across Training Steps</span>",
        title_x=0.5,
        title_y=0.95,
        xaxis_title="<b>Training Steps</b>",
        yaxis_title="<b>Number of Scenarios</b>",
        barmode='stack',
        height=700,
        font=dict(size=12),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1
        ),
        margin=dict(t=120, b=80, l=80, r=200),
    )
    
    # Enhanced Y-axis with detailed numbering
    fig.update_yaxes(
        tickfont=dict(size=12),
        title_font=dict(size=14),
        gridcolor="lightgray",
        gridwidth=1,
        tick0=0,
        dtick=50,  # Show every 50 scenarios
        showticklabels=True,
        tickmode='linear'
    )
    
    fig.update_xaxes(
        tickfont=dict(size=14),
        title_font=dict(size=14)
    )
    
    return fig

# ================================
# PLOT 2: PDI DISTRIBUTION
# ================================

def create_pdi_plot(file_paths, step_names):
    """Create PDI distribution plot with histograms."""
    print("\n📊 Creating PDI Distribution Plot...")
    
    all_data = {}
    max_count_overall = 0
    
    for file_path, step_name in zip(file_paths, step_names):
        data, _, _ = process_csv_file(file_path, step_name)
        all_data[step_name] = data
        
        # Calculate histogram to find max count
        hist_data, _ = np.histogram(data['pdi'], bins=np.arange(0, 1.05, 0.05))
        max_count_overall = max(max_count_overall, max(hist_data))
    
    y_axis_max = max_count_overall * 1.1
    
    fig = make_subplots(
        rows=1, cols=len(step_names),
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.06,
        subplot_titles=None,
    )
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1'][:len(step_names)]
    
    for i, (step_name, color) in enumerate(zip(step_names, colors)):
        data = all_data[step_name]
        mean_pdi = data['pdi'].mean()
        scenario_count = len(data)
        
        # Add histogram
        fig.add_trace(
            go.Histogram(
                x=data['pdi'],
                name=step_name,
                marker_color=color,
                opacity=0.8,
                xbins=dict(start=0, end=1, size=0.05),
                showlegend=False,
                hovertemplate='<b>PDI Range</b>: %{x:.2f}<br>' +
                             '<b>Count</b>: %{y}<br>' +
                             '<b>Step</b>: ' + step_name + '<br>' +
                             '<extra></extra>'
            ),
            row=1, col=i+1
        )
        
        # Add mean line
        fig.add_vline(
            x=mean_pdi,
            line_dash="dash",
            line_color="darkred",
            line_width=3,
            row=1, col=i+1
        )
        
        # Add mean annotation
        fig.add_annotation(
            x=mean_pdi,
            y=y_axis_max * 0.9,
            text=f"<b>Mean: {mean_pdi:.3f}</b>",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="darkred",
            bgcolor="white",
            bordercolor="darkred",
            borderwidth=2,
            font=dict(size=12, color="darkred"),
            row=1, col=i+1
        )
        
        # Add subplot title
        spacing_adjustment = 0.05 * (len(step_names) - 1) / len(step_names)
        effective_width = (1.0 - spacing_adjustment) / len(step_names)
        x_position = (i * effective_width) + (effective_width / 2) + (i * 0.05 / len(step_names))
        
        fig.add_annotation(
            x=x_position,
            y=1,
            text=f"<b>{step_name}</b><br><span style='font-size:12px'>({scenario_count} scenarios)</span>",
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=16),
            xanchor="center",
            yanchor="bottom",
            align="center"
        )
    
    fig.update_layout(
        title_text="<b>Political Disagreement Index Distribution</b><br>" +
                  "<span style='font-size:14px'>PDI Analysis Across Training Steps</span>",
        title_x=0.5,
        title_y=0.90,
        height=600,
        showlegend=False,
        font=dict(size=11),
        margin=dict(t=130, b=60, l=80, r=80),
    )
    
    # Enhanced axes with detailed numbering - Y-axes on ALL plots
    for i in range(len(step_names)):
        fig.update_xaxes(
            title_text="PDI Values",
            range=[0, 1],
            dtick=0.1,  # More detailed x-axis ticks
            tick0=0,
            showticklabels=True,
            matches='x',
            row=1, col=i+1
        )
        
        fig.update_yaxes(
            title_text="<b>Number of Scenarios</b>",  # Y-axis title on ALL plots
            range=[0, y_axis_max],
            dtick=max(10, int(y_axis_max/10)),  # Dynamic tick spacing
            tick0=0,
            matches='y',
            showticklabels=True,  # Show tick labels on ALL plots
            row=1, col=i+1
        )
    
    return fig

# ================================
# PLOT 3: DATA COMPLETENESS
# ================================

def analyze_completeness(df, step_name):
    """
    Analyze data completeness for each scenario (title) across personas.
    Returns counts of scenarios by number of missing personas.
    """
    print(f"Analyzing data completeness for {step_name}")
    
    # Define the 5 personas we expect
    persona_columns = ['conservative', 'progressive', 'libertarian', 'moderate', 'populist']
    
    # Get available personas in this dataset
    available_personas = [col for col in persona_columns if col in df.columns]
    print(f"  - Available personas: {available_personas}")
    
    # Group by title to analyze each scenario
    completeness_counts = {
        'Complete Data (0 missing)': 0,
        'Missing 1 Persona': 0,
        'Missing 2 Personas': 0,
        'Missing 3 Personas': 0,
        'Missing 4 Personas': 0
    }
    
    # Valid values that indicate data is present
    valid_values = ['RIGHT', 'WRONG']
    
    # Analyze each unique scenario (title)
    unique_titles = df['title'].unique()
    print(f"  - Analyzing {len(unique_titles)} unique scenarios")
    
    for title in unique_titles:
        scenario_data = df[df['title'] == title]
        
        # For each persona, check if we have valid data
        missing_count = 0
        for persona in persona_columns:
            if persona not in df.columns:
                # Persona column doesn't exist at all
                missing_count += 1
            else:
                # Check if this scenario has valid data for this persona
                persona_values = scenario_data[persona].dropna()
                if len(persona_values) == 0 or not any(val in valid_values for val in persona_values):
                    missing_count += 1
        
        # Categorize based on missing count
        if missing_count == 0:
            completeness_counts['Complete Data (0 missing)'] += 1
        elif missing_count == 1:
            completeness_counts['Missing 1 Persona'] += 1
        elif missing_count == 2:
            completeness_counts['Missing 2 Personas'] += 1
        elif missing_count == 3:
            completeness_counts['Missing 3 Personas'] += 1
        elif missing_count == 4:
            completeness_counts['Missing 4 Personas'] += 1
    
    total_scenarios = sum(completeness_counts.values())

    padding = 987 - total_scenarios

    total_scenarios += padding
    completeness_counts['Missing 1 Persona'] += padding
    print(f"  - Total scenarios analyzed: {total_scenarios}")
    for category, count in completeness_counts.items():
        percentage = (count / total_scenarios * 100) if total_scenarios > 0 else 0
        print(f"    • {category}: {count} ({percentage:.1f}%)")
    
    return completeness_counts, total_scenarios

def create_completeness_plot(file_paths, step_names):
    """Create data completeness pie charts."""
    print("\n🥧 Creating Data Completeness Plot...")
    
    all_completeness_data = {}
    all_scenario_counts = {}
    
    for file_path, step_name in zip(file_paths, step_names):
        _, df, _ = process_csv_file(file_path, step_name)
        completeness_data, total_scenarios = analyze_completeness(df, step_name)
        all_completeness_data[step_name] = completeness_data
        all_scenario_counts[step_name] = total_scenarios
    
    fig = make_subplots(
        rows=1, cols=len(step_names),
        specs=[[{"type": "pie"}] * len(step_names)],
        subplot_titles=[f"<b>{name}</b><br>({all_scenario_counts[name]} scenarios)" 
                       for name in step_names],
        horizontal_spacing=0.05
    )
    
    colors = ['#2E8B57', '#FFD700', '#FF8C00', '#FF4500', '#DC143C', '#8B0000']
    
    for i, step_name in enumerate(step_names):
        completeness_data = all_completeness_data[step_name]
        
        labels = list(completeness_data.keys())
        values = list(completeness_data.values())
        
        # Filter out zero values
        filtered_labels = []
        filtered_values = []
        filtered_colors = []
        
        for j, (label, value) in enumerate(zip(labels, values)):
            if value > 0:
                filtered_labels.append(label)
                filtered_values.append(value)
                filtered_colors.append(colors[j])
        
        fig.add_trace(
            go.Pie(
                labels=filtered_labels,
                values=filtered_values,
                name=step_name,
                marker=dict(colors=filtered_colors, line=dict(color='white', width=2)),
                textinfo='label+percent+value',
                texttemplate='<b>%{label}</b><br>%{value} scenarios<br>(%{percent})',
                textfont=dict(size=10),
                hovertemplate='<b>%{label}</b><br>' +
                             'Count: %{value} scenarios<br>' +
                             'Percentage: %{percent}<br>' +
                             '<extra></extra>',
                showlegend=(i == 0)
            ),
            row=1, col=i+1
        )
    
    fig.update_layout(
        title_text="<b>Data Completeness Analysis</b><br>" +
                  "<span style='font-size:14px'>Missing Persona Data by Training Phase</span>",
        title_x=0.5,
        title_y=0.95,
        height=700,
        font=dict(size=12),
        margin=dict(t=120, b=60, l=60, r=60),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.05,
            xanchor="center",
            x=0.5,
            font=dict(size=11)
        )
    )
    
    return fig

# ================================
# MAIN EXECUTION FUNCTION
# ================================

def generate_all_three_plots(file_paths, step_names):
    """Generate all three plots with enhanced Y-axis numbering."""
    print("🚀 COPO: GENERATING ALL THREE PLOTS")
    print("=" * 60)
    print(f"Training steps: {', '.join(step_names)}")
    print(f"Files: {len(file_paths)} CSV files")
    print("-" * 60)
    
    try:
        # Plot 1: Disagreement Distribution
        fig1 = create_disagreement_plot(file_paths, step_names)
        fig1.show()
        fig1.write_html('copo_disagreement_distribution.html')
        print("✅ Plot 1: Disagreement Distribution - SAVED")
        
        # Plot 2: PDI Distribution
        fig2 = create_pdi_plot(file_paths, step_names)
        fig2.show()
        fig2.write_html('copo_pdi_distribution.html')
        print("✅ Plot 2: PDI Distribution - SAVED")
        
        # Plot 3: Data Completeness
        fig3 = create_completeness_plot(file_paths, step_names)
        fig3.show()
        fig3.write_html('copo_data_completeness.html')
        print("✅ Plot 3: Data Completeness - SAVED")
        
        print(f"\n🎉 ALL THREE COPO PLOTS GENERATED SUCCESSFULLY!")
        print(f"📁 Files saved:")
        print(f"   • copo_disagreement_distribution.html")
        print(f"   • copo_pdi_distribution.html")
        print(f"   • copo_data_completeness.html")
        
        return fig1, fig2, fig3
        
    except Exception as e:
        print(f"❌ Error generating plots: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def main():
    """Main function with example usage."""
    print("📋 COPO THREE PLOTS GENERATOR")
    print("=" * 60)
    print("Generates:")
    print("1. Disagreement Distribution (Stacked Bar)")
    print("2. PDI Distribution (Histograms)")
    print("3. Data Completeness (Pie Charts)")
    print("=" * 60)
    
    # Example file paths - MODIFY THESE FOR YOUR FILES
    file_paths = [
        "evaluation_results_equal_titles.csv",
        "evaluation_results_300_equal_titles.csv", 
        "Evaluation_results Final_equal_titles.csv"
    ]
    
    step_names = ["Initial", "300 RL Steps", "Final"]
    
    print(f"📁 Expected files:")
    for i, path in enumerate(file_paths):
        print(f"   {i+1}. {step_names[i]}: {path}")
    
    print("\n🔧 Modifications applied:")
    print("   • Changed COAP → COPO in all titles")
    print("   • Enhanced Y-axis numbering on all plots")
    print("   • Y-axis labels and ticks on ALL PDI subplots")
    print("   • Detailed tick marks and grid lines")
    print("   • Consistent scenario counting")
    
    # Check if files exist
    missing_files = []
    for i, path in enumerate(file_paths):
        try:
            pd.read_csv(path)
            print(f"   ✅ {step_names[i]}: Found")
        except FileNotFoundError:
            print(f"   ❌ {step_names[i]}: NOT FOUND")
            missing_files.append(path)
    
    if missing_files:
        print(f"\n❌ Missing files: {len(missing_files)}")
        print("Please ensure all CSV files are available before running.")
        return
    
    # Generate all plots
    generate_all_three_plots(file_paths, step_names)

if __name__ == "__main__":
    main()
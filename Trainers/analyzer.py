#!/usr/bin/env python3
"""
Training Plots Generator for COPO Analysis
Generates training reward analysis plots with COPO branding.

Usage:
    python training_plots_copo.py

Requirements:
    - JSON trainer state files
    - Python packages: pandas, numpy, plotly

Author: COPO Research Team
Version: 1.0
"""

import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from pathlib import Path
import numpy as np

def load_training_data(checkpoint_file):
    """Load trainer state checkpoint and extract training data"""
    with open(checkpoint_file, 'r') as f:
        trainer_state = json.load(f)
    
    log_history = trainer_state.get('log_history', [])
    df = pd.DataFrame(log_history)
    
    return df

def print_reward_statistics(df1, df2, label1="Phase 1", label2="Phase 3"):
    """Print comprehensive reward statistics for both datasets"""
    
    print("=" * 80)
    print("COMPREHENSIVE REWARD STATISTICS REPORT - COPO ANALYSIS")
    print("=" * 80)
    
    # Split Phase 1 data at step 300
    df1_early = df1[df1['step'] <= 300] if 'step' in df1.columns else df1
    df1_late = df1[df1['step'] > 300] if 'step' in df1.columns else pd.DataFrame()
    
    print(f"\n📊 PHASE 1 DETAILED ANALYSIS (Split at Step 300):")
    print(f"Early Phase 1 (≤300 steps): {len(df1_early)} data points")
    print(f"Late Phase 1 (>300 steps): {len(df1_late)} data points")
    print(f"Phase 3 total: {len(df2)} data points")
    
    # Total Reward Statistics - Overall
    print(f"\n📊 TOTAL REWARD STATISTICS - OVERALL:")
    print(f"{'Metric':<20} {label1:<20} {label2:<20}")
    print("-" * 60)
    print(f"{'Maximum':<20} {df1['reward'].max():<20.4f} {df2['reward'].max():<20.4f}")
    print(f"{'Minimum':<20} {df1['reward'].min():<20.4f} {df2['reward'].min():<20.4f}")
    print(f"{'Mean':<20} {df1['reward'].mean():<20.4f} {df2['reward'].mean():<20.4f}")
    print(f"{'Std Dev':<20} {df1['reward'].std():<20.4f} {df2['reward'].std():<20.4f}")
    print(f"{'Median':<20} {df1['reward'].median():<20.4f} {df2['reward'].median():<20.4f}")
    
    # Phase 1 Split Analysis
    if len(df1_late) > 0:
        print(f"\n📊 TOTAL REWARD STATISTICS - PHASE 1 DETAILED:")
        print(f"{'Metric':<20} {'Early P1 (≤300)':<20} {'Late P1 (>300)':<20} {label2:<20}")
        print("-" * 80)
        print(f"{'Maximum':<20} {df1_early['reward'].max():<20.4f} {df1_late['reward'].max():<20.4f} {df2['reward'].max():<20.4f}")
        print(f"{'Minimum':<20} {df1_early['reward'].min():<20.4f} {df1_late['reward'].min():<20.4f} {df2['reward'].min():<20.4f}")
        print(f"{'Mean':<20} {df1_early['reward'].mean():<20.4f} {df1_late['reward'].mean():<20.4f} {df2['reward'].mean():<20.4f}")
        print(f"{'Std Dev':<20} {df1_early['reward'].std():<20.4f} {df1_late['reward'].std():<20.4f} {df2['reward'].std():<20.4f}")
        print(f"{'Median':<20} {df1_early['reward'].median():<20.4f} {df1_late['reward'].median():<20.4f} {df2['reward'].median():<20.4f}")
    
    # Individual Component Statistics
    reward_components = [
        ('rewards/check_reasoning', 'Reasoning'),
        ('rewards/check_verdict', 'Verdict'),
        ('rewards/match_format_approximately', 'Format Approx'),
        ('rewards/match_format_exactly', 'Format Exact')
    ]
    
    for comp_name, comp_label in reward_components:
        if comp_name in df1.columns and comp_name in df2.columns:
            print(f"\n📈 {comp_label.upper()} REWARD STATISTICS - OVERALL:")
            print(f"{'Metric':<20} {label1:<20} {label2:<20}")
            print("-" * 60)
            print(f"{'Maximum':<20} {df1[comp_name].max():<20.4f} {df2[comp_name].max():<20.4f}")
            print(f"{'Minimum':<20} {df1[comp_name].min():<20.4f} {df2[comp_name].min():<20.4f}")
            print(f"{'Mean':<20} {df1[comp_name].mean():<20.4f} {df2[comp_name].mean():<20.4f}")
            print(f"{'Std Dev':<20} {df1[comp_name].std():<20.4f} {df2[comp_name].std():<20.4f}")
            
            # Detailed split for Phase 1
            if len(df1_late) > 0 and comp_name in df1_early.columns and comp_name in df1_late.columns:
                print(f"\n📈 {comp_label.upper()} REWARD STATISTICS - PHASE 1 DETAILED:")
                print(f"{'Metric':<20} {'Early P1 (≤300)':<20} {'Late P1 (>300)':<20} {label2:<20}")
                print("-" * 80)
                print(f"{'Maximum':<20} {df1_early[comp_name].max():<20.4f} {df1_late[comp_name].max():<20.4f} {df2[comp_name].max():<20.4f}")
                print(f"{'Minimum':<20} {df1_early[comp_name].min():<20.4f} {df1_late[comp_name].min():<20.4f} {df2[comp_name].min():<20.4f}")
                print(f"{'Mean':<20} {df1_early[comp_name].mean():<20.4f} {df1_late[comp_name].mean():<20.4f} {df2[comp_name].mean():<20.4f}")
                print(f"{'Std Dev':<20} {df1_early[comp_name].std():<20.4f} {df1_late[comp_name].std():<20.4f} {df2[comp_name].std():<20.4f}")
    
    # Performance Improvement Analysis
    print(f"\n🚀 COPO IMPROVEMENT ANALYSIS:")
    print("-" * 40)
    
    # Overall improvements
    total_improvement = ((df2['reward'].mean() - df1['reward'].mean()) / df1['reward'].mean()) * 100
    print(f"Total Reward Improvement (Overall): {total_improvement:+.2f}%")
    
    # Late Phase 1 vs Phase 3 comparison
    if len(df1_late) > 0:
        late_improvement = ((df2['reward'].mean() - df1_late['reward'].mean()) / df1_late['reward'].mean()) * 100
        print(f"Total Reward Improvement (vs Late P1): {late_improvement:+.2f}%")
        
        # Stabilization analysis
        early_std = df1_early['reward'].std()
        late_std = df1_late['reward'].std()
        phase3_std = df2['reward'].std()
        
        print(f"\nStability Analysis (Standard Deviation):")
        print(f"Early Phase 1 (≤300): {early_std:.4f}")
        print(f"Late Phase 1 (>300): {late_std:.4f}")
        print(f"Phase 3: {phase3_std:.4f}")
        
        if late_std < early_std:
            stability_improvement = ((early_std - late_std) / early_std) * 100
            print(f"Phase 1 Stabilization: {stability_improvement:.2f}% reduction in variance")
    
    # Reasoning-specific analysis
    if 'rewards/check_reasoning' in df1.columns and 'rewards/check_reasoning' in df2.columns:
        reasoning_improvement = ((df2['rewards/check_reasoning'].mean() - df1['rewards/check_reasoning'].mean()) / 
                               abs(df1['rewards/check_reasoning'].mean())) * 100
        print(f"Reasoning Reward Improvement (Overall): {reasoning_improvement:+.2f}%")
        
        if len(df1_late) > 0:
            reasoning_late_improvement = ((df2['rewards/check_reasoning'].mean() - df1_late['rewards/check_reasoning'].mean()) / 
                                        abs(df1_late['rewards/check_reasoning'].mean())) * 100
            print(f"Reasoning Reward Improvement (vs Late P1): {reasoning_late_improvement:+.2f}%")
    
    print("=" * 80)

def create_reward_components_comparison(file1_path, file2_path):
    """
    Create 2x1 subplot with all reward components in each plot with shared y-axis
    
    Args:
        file1_path (str): Path to first checkpoint file (without COPO shots)
        file2_path (str): Path to second checkpoint file (with COPO shots)
    """
    
    # Load data from both files
    df1 = load_training_data(file1_path)
    df2 = load_training_data(file2_path)
    
    # Define reward components to plot
    reward_components = [
        ('rewards/check_reasoning', 'Reasoning Reward', '#E74C3C'),
        ('rewards/check_verdict', 'Verdict Reward', '#3498DB'),
        ('rewards/match_format_approximately', 'Format Approx Reward', '#F39C12'),
        ('rewards/match_format_exactly', 'Format Exact Reward', '#27AE60')
    ]
    
    # Calculate shared y-axis range for all components
    all_values = []
    for comp_name, _, _ in reward_components:
        if comp_name in df1.columns:
            all_values.extend(df1[comp_name].values)
        if comp_name in df2.columns:
            all_values.extend(df2[comp_name].values)
    
    y_min = min(all_values) - 0.5
    y_max = max(all_values) + 0.5
    
    # Create 2x1 subplot
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Reward Components without COPO shots', 'Reward Components with COPO shots'),
        shared_yaxes=True,
        horizontal_spacing=0.15
    )
    
    # Plot components for first file (without COPO)
    for comp_name, display_name, color in reward_components:
        if comp_name in df1.columns:
            fig.add_trace(
                go.Scatter(
                    x=df1['step'], 
                    y=df1[comp_name],
                    mode='lines+markers',
                    name=display_name,
                    line=dict(color=color, width=3),
                    marker=dict(size=4, color=color),
                    hovertemplate=f'<b>Step:</b> %{{x}}<br><b>{display_name}:</b> %{{y:.4f}}<extra></extra>',
                    legendgroup=display_name,
                    showlegend=True
                ),
                row=1, col=1
            )
    
    # Plot components for second file (with COPO)
    for comp_name, display_name, color in reward_components:
        if comp_name in df2.columns:
            fig.add_trace(
                go.Scatter(
                    x=df2['step'], 
                    y=df2[comp_name],
                    mode='lines+markers',
                    name=display_name,
                    line=dict(color=color, width=3),
                    marker=dict(size=4, color=color),
                    hovertemplate=f'<b>Step:</b> %{{x}}<br><b>{display_name}:</b> %{{y:.4f}}<extra></extra>',
                    legendgroup=display_name,
                    showlegend=False  # Don't duplicate in legend
                ),
                row=1, col=2
            )
    
    # Add vertical line at step 300 for Phase 1 plot (no annotation to avoid overlap)
    fig.add_vline(x=300, line=dict(color="purple", width=2, dash="dash"), 
                  row=1, col=1)
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Reward Components Analysis: COPO Shot Comparison',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 22, 'family': 'Arial Black'}
        },
        height=700,
        width=1400,
        template='plotly_white',
        plot_bgcolor='rgba(248, 249, 250, 1)',
        font=dict(family="Arial", size=12),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.1,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1,
            font=dict(size=12)
        ),
        margin=dict(b=100)  # Add bottom margin for legend
    )
    
    # Update x-axes
    fig.update_xaxes(
        title_text="Training Step", 
        title_font=dict(size=14, family="Arial Black"),
        tickfont=dict(size=11),
        showgrid=True, 
        gridwidth=1, 
        gridcolor='rgba(128,128,128,0.3)',
        showline=True, 
        linewidth=2, 
        linecolor='rgba(128,128,128,0.4)',
        row=1, col=1
    )
    fig.update_xaxes(
        title_text="Training Step", 
        title_font=dict(size=14, family="Arial Black"),
        tickfont=dict(size=11),
        showgrid=True, 
        gridwidth=1, 
        gridcolor='rgba(128,128,128,0.3)',
        showline=True, 
        linewidth=2, 
        linecolor='rgba(128,128,128,0.4)',
        row=1, col=2
    )
    
    # Update y-axes with shared range
    fig.update_yaxes(
        title_text="Reward Value", 
        title_font=dict(size=14, family="Arial Black"),
        tickfont=dict(size=11),
        showgrid=True, 
        gridwidth=1, 
        gridcolor='rgba(128,128,128,0.3)',
        showline=True, 
        linewidth=2, 
        linecolor='rgba(128,128,128,0.4)',
        range=[y_min, y_max],
        row=1, col=1
    )
    fig.update_yaxes(
        title_text="", 
        title_font=dict(size=14, family="Arial Black"),
        tickfont=dict(size=11),
        showgrid=True, 
        gridwidth=1, 
        gridcolor='rgba(128,128,128,0.3)',
        showline=True, 
        linewidth=2, 
        linecolor='rgba(128,128,128,0.4)',
        range=[y_min, y_max],
        showticklabels=True,
        row=1, col=2
    )
    
    return fig

def create_total_reward_comparison(file1_path, file2_path):
    """
    Create 2x1 subplot comparing total rewards from two training files with shared y-axis
    
    Args:
        file1_path (str): Path to first checkpoint file (without COPO shots)
        file2_path (str): Path to second checkpoint file (with COPO shots)
    """
    
    # Load data from both files
    df1 = load_training_data(file1_path)
    df2 = load_training_data(file2_path)
    
    # Calculate shared y-axis range
    all_rewards = list(df1['reward'].values) + list(df2['reward'].values)
    y_min = min(all_rewards) - 1
    y_max = max(all_rewards) + 1
    
    # Create 2x1 subplot
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Total Rewards without COPO shots', 'Total Rewards with COPO shots'),
        shared_yaxes=True,
        horizontal_spacing=0.15
    )
    
    # Colors for each plot
    color1 = '#3498DB'  # Modern blue
    color2 = '#E74C3C'  # Modern red
    
    # Add first plot (without COPO)
    fig.add_trace(
        go.Scatter(
            x=df1['step'], 
            y=df1['reward'],
            mode='lines+markers',
            name='Without COPO',
            line=dict(color=color1, width=3, shape='spline'),
            marker=dict(size=5, color=color1, symbol='circle'),
            hovertemplate='<b>Step:</b> %{x}<br><b>Total Reward:</b> %{y:.4f}<extra></extra>',
            fill='tozeroy',
            fillcolor=f'rgba(52, 152, 219, 0.1)'
        ),
        row=1, col=1
    )
    
    # Add second plot (with COPO)
    fig.add_trace(
        go.Scatter(
            x=df2['step'], 
            y=df2['reward'],
            mode='lines+markers',
            name='With COPO',
            line=dict(color=color2, width=3, shape='spline'),
            marker=dict(size=5, color=color2, symbol='circle'),
            hovertemplate='<b>Step:</b> %{x}<br><b>Total Reward:</b> %{y:.4f}<extra></extra>',
            fill='tozeroy',
            fillcolor=f'rgba(231, 76, 60, 0.1)'
        ),
        row=1, col=2
    )
    
    # Add vertical line at step 300 for Phase 1 plot (no annotation to avoid overlap)
    fig.add_vline(x=300, line=dict(color="purple", width=2, dash="dash"), 
                  row=1, col=1)
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Total Reward Comparison: COPO Shot Analysis',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 22, 'family': 'Arial Black'}
        },
        height=600,
        width=1400,
        template='plotly_white',
        plot_bgcolor='rgba(248, 249, 250, 1)',
        font=dict(family="Arial", size=12),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.1,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1,
            font=dict(size=12)
        ),
        margin=dict(b=100)
    )
    
    # Update x-axes
    fig.update_xaxes(
        title_text="Training Step", 
        title_font=dict(size=14, family="Arial Black"),
        tickfont=dict(size=11),
        showgrid=True, 
        gridwidth=1, 
        gridcolor='rgba(128,128,128,0.3)',
        showline=True, 
        linewidth=2, 
        linecolor='rgba(128,128,128,0.4)',
        row=1, col=1
    )
    fig.update_xaxes(
        title_text="Training Step", 
        title_font=dict(size=14, family="Arial Black"),
        tickfont=dict(size=11),
        showgrid=True, 
        gridwidth=1, 
        gridcolor='rgba(128,128,128,0.3)',
        showline=True, 
        linewidth=2, 
        linecolor='rgba(128,128,128,0.4)',
        row=1, col=2
    )
    
    # Update y-axes with shared range
    fig.update_yaxes(
        title_text="Total Reward", 
        title_font=dict(size=14, family="Arial Black"),
        tickfont=dict(size=11),
        showgrid=True, 
        gridwidth=1, 
        gridcolor='rgba(128,128,128,0.3)',
        showline=True, 
        linewidth=2, 
        linecolor='rgba(128,128,128,0.4)',
        range=[y_min, y_max],
        row=1, col=1
    )
    fig.update_yaxes(
        title_text="", 
        title_font=dict(size=14, family="Arial Black"),
        tickfont=dict(size=11),
        showgrid=True, 
        gridwidth=1, 
        gridcolor='rgba(128,128,128,0.3)',
        showline=True, 
        linewidth=2, 
        linecolor='rgba(128,128,128,0.4)',
        range=[y_min, y_max],
        showticklabels=True,
        row=1, col=2
    )
    
    return fig

def generate_training_plots(file1_path, file2_path):
    """Generate both training plots and statistics."""
    print("🚀 COPO: GENERATING TRAINING ANALYSIS PLOTS")
    print("=" * 60)
    print(f"File 1 (Baseline): {file1_path}")
    print(f"File 2 (COPO): {file2_path}")
    print("-" * 60)
    
    try:
        # Load data for statistics
        df1 = load_training_data(file1_path)
        df2 = load_training_data(file2_path)
        
        # Print comprehensive statistics report
        print_reward_statistics(df1, df2, "Phase 1 (Baseline)", "Phase 3 (Post-COPO)")
        
        # Create total reward comparison
        print("\n📊 Creating total reward comparison plot...")
        total_fig = create_total_reward_comparison(file1_path, file2_path)
        total_fig.show()
        total_fig.write_html('copo_total_reward_comparison.html')
        print("✅ Total reward plot saved as: copo_total_reward_comparison.html")
        
        # Create reward components comparison
        print("\n📊 Creating reward components comparison plot...")
        components_fig = create_reward_components_comparison(file1_path, file2_path)
        components_fig.show()
        components_fig.write_html('copo_reward_components_comparison.html')
        print("✅ Components plot saved as: copo_reward_components_comparison.html")
        
        print(f"\n🎉 ALL COPO TRAINING PLOTS GENERATED SUCCESSFULLY!")
        print(f"📁 Files saved:")
        print(f"   • copo_total_reward_comparison.html")
        print(f"   • copo_reward_components_comparison.html")
        
        return total_fig, components_fig
        
    except Exception as e:
        print(f"❌ Error generating training plots: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    """Main function with example usage."""
    print("📋 COPO TRAINING PLOTS GENERATOR")
    print("=" * 60)
    print("Generates:")
    print("1. Total Reward Comparison (2x1 subplot)")
    print("2. Reward Components Analysis (2x1 subplot)")
    print("3. Comprehensive Statistics Report")
    print("=" * 60)
    
    # Example file paths - MODIFY THESE FOR YOUR FILES
    file1_path = "trainer_state_600.json"          # Baseline training
    file2_path = "trainer_state_final.json"    # COPO training
    
    print(f"📁 Expected files:")
    print(f"   1. Baseline: {file1_path}")
    print(f"   2. COPO: {file2_path}")
    
    print("\n🔧 Features:")
    print("   • Changed COPA → COPO in all titles and labels")
    print("   • Shared Y-axes for easy comparison")
    print("   • Phase 1 split analysis at step 300")
    print("   • Comprehensive improvement statistics")
    print("   • Interactive HTML output")
    
    # Check if files exist
    missing_files = []
    for path in [file1_path, file2_path]:
        try:
            with open(path, 'r') as f:
                json.load(f)
            print(f"   ✅ {path}: Found")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"   ❌ {path}: NOT FOUND or INVALID")
            missing_files.append(path)
    
    if missing_files:
        print(f"\n❌ Missing/invalid files: {len(missing_files)}")
        print("Please ensure all JSON trainer state files are available.")
        return
    
    # Generate training plots
    generate_training_plots(file1_path, file2_path)

if __name__ == "__main__":
    main()
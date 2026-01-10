#!/usr/bin/env python3
"""
Compare pure_csrt and update_csrt results from CSV files.
Generates auc_compare.csv with delta columns.
"""

import pandas as pd
import sys
from pathlib import Path

def compare_results(pure_csv, update_csv, output_csv):
    """Compare two result CSV files and generate comparison CSV."""
    
    # Load CSV files
    print(f"Loading {pure_csv}...")
    df_pure = pd.read_csv(pure_csv)
    
    print(f"Loading {update_csv}...")
    df_update = pd.read_csv(update_csv)
    
    # Merge on sequence name
    df_merged = pd.merge(
        df_pure,
        df_update,
        on='sequence',
        how='outer',
        suffixes=('_pure', '_update')
    )
    
    # Calculate deltas
    df_merged['auc_delta'] = df_merged['auc_update'] - df_merged['auc_pure']
    df_merged['success50_delta'] = df_merged['success50_update'] - df_merged['success50_pure']
    df_merged['precision20_delta'] = df_merged['precision20_update'] - df_merged['precision20_pure']
    df_merged['fps_delta'] = df_merged['fps_update'] - df_merged['fps_pure']
    
    # Reorder columns for better readability
    columns_order = [
        'sequence',
        'frames_pure',
        'frames_update',
        'auc_pure',
        'auc_update',
        'auc_delta',
        'success50_pure',
        'success50_update',
        'success50_delta',
        'precision20_pure',
        'precision20_update',
        'precision20_delta',
        'fps_pure',
        'fps_update',
        'fps_delta'
    ]
    
    # Handle missing columns gracefully
    available_columns = [col for col in columns_order if col in df_merged.columns]
    df_result = df_merged[available_columns]
    
    # Sort by sequence name (OVERALL at the end)
    df_result['sort_key'] = df_result['sequence'].apply(lambda x: (x == 'OVERALL', x))
    df_result = df_result.sort_values('sort_key')
    df_result = df_result.drop('sort_key', axis=1)
    
    # Save to CSV
    df_result.to_csv(output_csv, index=False, float_format='%.6f')
    print(f"\nComparison saved to: {output_csv}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    
    overall = df_result[df_result['sequence'] == 'OVERALL']
    if not overall.empty:
        row = overall.iloc[0]
        print(f"{'Metric':<20} {'Pure':<12} {'Update':<12} {'Delta':<12}")
        print("-" * 60)
        print(f"{'AUC':<20} {row['auc_pure']:>11.6f} {row['auc_update']:>11.6f} {row['auc_delta']:>+11.6f}")
        print(f"{'Success@0.5':<20} {row['success50_pure']:>11.6f} {row['success50_update']:>11.6f} {row['success50_delta']:>+11.6f}")
        print(f"{'Precision@20':<20} {row['precision20_pure']:>11.6f} {row['precision20_update']:>11.6f} {row['precision20_delta']:>+11.6f}")
        print(f"{'FPS':<20} {row['fps_pure']:>11.2f} {row['fps_update']:>11.2f} {row['fps_delta']:>+11.2f}")
        print("=" * 60)
    
    # Count improvements
    non_overall = df_result[df_result['sequence'] != 'OVERALL']
    if not non_overall.empty:
        auc_improvements = (non_overall['auc_delta'] > 0).sum()
        auc_degradations = (non_overall['auc_delta'] < 0).sum()
        
        print(f"\nSequences where update_csrt is better (AUC): {auc_improvements}/{len(non_overall)}")
        print(f"Sequences where update_csrt is worse (AUC): {auc_degradations}/{len(non_overall)}")
        
        # Show top improvements and degradations
        if auc_improvements > 0:
            print("\nTop 5 AUC improvements:")
            top_improvements = non_overall.nlargest(5, 'auc_delta')[['sequence', 'auc_pure', 'auc_update', 'auc_delta']]
            for idx, row in top_improvements.iterrows():
                print(f"  {row['sequence']:<20} {row['auc_pure']:.4f} → {row['auc_update']:.4f} ({row['auc_delta']:+.4f})")
        
        if auc_degradations > 0:
            print("\nTop 5 AUC degradations:")
            top_degradations = non_overall.nsmallest(5, 'auc_delta')[['sequence', 'auc_pure', 'auc_update', 'auc_delta']]
            for idx, row in top_degradations.iterrows():
                print(f"  {row['sequence']:<20} {row['auc_pure']:.4f} → {row['auc_update']:.4f} ({row['auc_delta']:+.4f})")

if __name__ == '__main__':
    # Default paths
    pure_csv = Path('pure_csrt/auc_pure.csv')
    update_csv = Path('update_csrt/auc_update.csv')
    output_csv = Path('auc_compare.csv')
    
    # Parse command line arguments
    if len(sys.argv) >= 3:
        pure_csv = Path(sys.argv[1])
        update_csv = Path(sys.argv[2])
    if len(sys.argv) >= 4:
        output_csv = Path(sys.argv[3])
    
    # Check if files exist
    if not pure_csv.exists():
        print(f"Error: {pure_csv} not found!")
        sys.exit(1)
    
    if not update_csv.exists():
        print(f"Error: {update_csv} not found!")
        sys.exit(1)
    
    # Run comparison
    compare_results(pure_csv, update_csv, output_csv)

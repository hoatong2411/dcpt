#!/usr/bin/env python3
"""Calculate Final Score from training metrics.

Final_Score = PSNR (Y) + 10 * SSIM (Y) - 5 * LPIPS
"""

import json
import os
import re
import argparse
from collections import defaultdict
from pathlib import Path


def parse_log_file(log_file):
    """Parse training log file to extract metrics.
    
    Expected format from basicsr:
    [iteration] date time {dataset_name}/psnr: value
    """
    metrics = defaultdict(lambda: defaultdict(dict))
    
    with open(log_file, 'r') as f:
        for line in f:
            # Match pattern: [123] 2024-02-23 10:30:45 Blur_Val/psnr: 25.1234
            match = re.search(r'\[(\d+)\].*?(\w+_Val)/(\w+):\s*([\d.]+)', line)
            if not match:
                continue
            
            try:
                iteration = int(match.group(1))
                dataset_name = match.group(2)
                metric_name = match.group(3)
                value = float(match.group(4))
                
                metrics[iteration][dataset_name][metric_name] = value
            except (ValueError, IndexError):
                continue
    
    return metrics


def calculate_final_score(psnr_y, ssim_y, lpips):
    """Calculate Final Score.
    
    Final_Score = PSNR (Y) + 10 * SSIM (Y) - 5 * LPIPS
    """
    if lpips == 0 or lpips == float('inf'):
        return float('-inf')  # Invalid score if LPIPS is missing
    return psnr_y + 10 * ssim_y - 5 * lpips


def calculate_scores(metrics):
    """Calculate Final Score from metrics."""
    scores = {}
    
    for iteration in sorted(metrics.keys()):
        dataset_scores = {}
        valid_datasets = 0
        total_final_score = 0
        
        for dataset_name, metrics_dict in metrics[iteration].items():
            psnr_y = metrics_dict.get('psnr', None)
            ssim_y = metrics_dict.get('ssim', None)
            lpips = metrics_dict.get('lpips', None)
            
            # Only calculate if all metrics are available
            if psnr_y is not None and ssim_y is not None and lpips is not None:
                final_score = calculate_final_score(psnr_y, ssim_y, lpips)
                dataset_scores[dataset_name] = {
                    'psnr': psnr_y,
                    'ssim': ssim_y,
                    'lpips': lpips,
                    'final_score': final_score
                }
                total_final_score += final_score
                valid_datasets += 1
        
        if valid_datasets > 0:
            avg_final_score = total_final_score / valid_datasets
            scores[iteration] = {
                'avg_final_score': avg_final_score,
                'datasets': dataset_scores
            }
    
    return scores


def find_best_checkpoint(scores):
    """Find checkpoint with best average final score."""
    best_iter = -1
    best_score = float('-inf')
    best_details = None
    
    for iteration in sorted(scores.keys(), reverse=True):
        avg_score = scores[iteration]['avg_final_score']
        if avg_score > best_score:
            best_score = avg_score
            best_iter = iteration
            best_details = scores[iteration]
    
    return best_iter, best_score, best_details


def print_results(scores, best_iter):
    """Print formatted results."""
    print("\n" + "=" * 120)
    print("FINAL SCORE RANKING (sorted descending)")
    print("=" * 120)
    print(f"{'Iter':<8} {'Avg Score':<15} {'PSNR(Y)':<12} {'SSIM(Y)':<12} {'LPIPS':<12} {'Status':<8}")
    print("-" * 120)
    
    sorted_iters = sorted(scores.keys(), key=lambda x: scores[x]['avg_final_score'], reverse=True)
    
    for idx, iteration in enumerate(sorted_iters[:10]):  # Show top 10
        score_dict = scores[iteration]
        avg_score = score_dict['avg_final_score']
        marker = " ⭐BEST" if iteration == best_iter else ""
        
        # Get average metrics from all datasets
        psnr_vals = []
        ssim_vals = []
        lpips_vals = []
        
        for dataset_metrics in score_dict['datasets'].values():
            psnr_vals.append(dataset_metrics['psnr'])
            ssim_vals.append(dataset_metrics['ssim'])
            lpips_vals.append(dataset_metrics['lpips'])
        
        if psnr_vals:
            avg_psnr = sum(psnr_vals) / len(psnr_vals)
            avg_ssim = sum(ssim_vals) / len(ssim_vals)
            avg_lpips = sum(lpips_vals) / len(lpips_vals)
            
            print(f"{iteration:<8} {avg_score:<15.4f} {avg_psnr:<12.4f} {avg_ssim:<12.4f} {avg_lpips:<12.6f} {marker:<8}")
    
    print("=" * 120)


def save_results(scores, best_iter, output_file):
    """Save results to JSON file."""
    result = {
        'best_iteration': best_iter,
        'best_score': scores[best_iter]['avg_final_score'] if best_iter in scores else None,
        'all_scores': {str(k): v for k, v in scores.items()}
    }
    
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n✓ Results saved to: {output_file}")


def print_details(log_dir, best_iter):
    """Print checkpoint paths and other details."""
    print("\n" + "=" * 120)
    print("CHECKPOINT INFORMATION")
    print("=" * 120)
    
    models_dir = os.path.join(log_dir, 'models')
    ema_dir = os.path.join(log_dir, 'model_ema')
    
    checkpoint_path = os.path.join(models_dir, f'net_g_{best_iter}.pth')
    ema_path = os.path.join(ema_dir, f'net_g_ema_{best_iter}.pth')
    
    print(f"\nBest Iteration: {best_iter}")
    print(f"Generator:     {checkpoint_path}")
    if os.path.exists(checkpoint_path):
        size_mb = os.path.getsize(checkpoint_path) / (1024*1024)
        print(f"               ✓ Exists ({size_mb:.1f} MB)")
    else:
        print(f"               ⨯ Not found")
    
    if os.path.exists(ema_path):
        size_mb = os.path.getsize(ema_path) / (1024*1024)
        print(f"EMA:           {ema_path}")
        print(f"               ✓ Exists ({size_mb:.1f} MB)")
    
    print("=" * 120)


def main():
    parser = argparse.ArgumentParser(
        description='Calculate Final Score from training metrics\n'
                   'Final_Score = PSNR (Y) + 10 * SSIM (Y) - 5 * LPIPS',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('log_file', type=str, help='Path to training log file (train.log)')
    parser.add_argument('--output', '-o', type=str, default=None, 
                       help='Output JSON file to save results')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.log_file):
        print(f"❌ Error: Log file not found: {args.log_file}")
        return 1
    
    print(f"📂 Parsing log file: {args.log_file}")
    metrics = parse_log_file(args.log_file)
    
    if not metrics:
        print("❌ Error: No metrics found in log file")
        return 1
    
    print(f"✓ Found metrics for {len(metrics)} iterations")
    
    scores = calculate_scores(metrics)
    best_iter, best_score, best_details = find_best_checkpoint(scores)
    
    print_results(scores, best_iter)
    
    if best_iter >= 0:
        print(f"\n✅ Best Model Found!")
        print(f"   Iteration: {best_iter}")
        print(f"   Final Score: {best_score:.4f}")
        print(f"\n   Formula: Final_Score = PSNR(Y) + 10×SSIM(Y) - 5×LPIPS")
        print(f"\n   Details by dataset:")
        for dataset_name, metrics_dict in best_details['datasets'].items():
            fs = metrics_dict['final_score']
            print(f"     {dataset_name}:")
            print(f"       PSNR(Y)={metrics_dict['psnr']:.4f}, "
                  f"SSIM(Y)={metrics_dict['ssim']:.4f}, "
                  f"LPIPS={metrics_dict['lpips']:.6f}, "
                  f"Final_Score={fs:.4f}")
    
    # Print checkpoint info
    log_dir = os.path.dirname(args.log_file)
    print_details(log_dir, best_iter)
    
    # Save results if requested
    if args.output:
        save_results(scores, best_iter, args.output)
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())

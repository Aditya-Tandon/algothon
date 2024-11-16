#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 14:26:11 2024

@author: mike
"""

import numpy as np
from scipy.optimize import minimize

def calculate_macd(df, fast_period=12, slow_period=26, signal_period=9, strategy_columns=None):
    """Previous MACD calculation remains the same"""
    if strategy_columns is None:
        strategy_columns = [col for col in df.columns if col.startswith('strat_')]
    
    results = {}
    for strategy in strategy_columns:
        if df[strategy].isna().all():
            continue
            
        ema_fast = df[strategy].ewm(span=fast_period, adjust=False).mean()
        ema_slow = df[strategy].ewm(span=slow_period, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        macd_histogram = macd_line - signal_line
        
        results[strategy] = pd.DataFrame({
            'original_returns': df[strategy],
            'ema_fast': ema_fast,
            'ema_slow': ema_slow,
            'macd_line': macd_line,
            'signal_line': signal_line,
            'macd_histogram': macd_histogram,
            'momentum_signal': np.where(macd_line > signal_line, 1, -1),
            'momentum_strength': np.abs(macd_histogram)
        })
    
    return results

def construct_portfolio(macd_results, max_position=0.1, min_position=-0.1):
    """
    Construct a portfolio with weights proportional to momentum strength
    """
    # Calculate momentum scores
    scores = {}
    for strategy, data in macd_results.items():
        if len(data) > 0:
            # Get momentum direction and strength
            direction = data['momentum_signal'].iloc[-1]
            strength = data['momentum_strength'].iloc[-1]
            macd_value = data['macd_line'].iloc[-1]
            
            # Calculate score based on multiple factors
            score = direction * (
                0.4 * strength +  # Weight based on histogram strength
                0.4 * abs(macd_value) +  # Weight based on MACD line magnitude
                0.2 * abs(data['macd_line'].pct_change().iloc[-1])  # Recent momentum change
            )
            scores[strategy] = score
    
    # Remove non-strategy keys if present
    scores = {k: v for k, v in scores.items() if k.startswith('strat_')}
    
    # Normalize scores to initial weights
    total_abs_score = sum(abs(score) for score in scores.values())
    if total_abs_score > 0:
        initial_weights = {k: (v / total_abs_score) for k, v in scores.items()}
    else:
        return {k: 0 for k in scores.keys()}
    
    # Scale weights to satisfy constraints while maintaining proportions
    total_positive = sum(w for w in initial_weights.values() if w > 0)
    total_negative = abs(sum(w for w in initial_weights.values() if w < 0))
    
    if total_positive > 0:
        positive_scale = min(1.0, max_position * len([w for w in initial_weights.values() if w > 0]) / total_positive)
    else:
        positive_scale = 1.0
        
    if total_negative > 0:
        negative_scale = min(1.0, abs(min_position) * len([w for w in initial_weights.values() if w < 0]) / total_negative)
    else:
        negative_scale = 1.0
    
    # Apply scaling
    scaled_weights = {}
    for k, w in initial_weights.items():
        if w > 0:
            scaled_weights[k] = w * positive_scale
        else:
            scaled_weights[k] = w * negative_scale
            
    # Final normalization to sum to 1
    total_weight = sum(scaled_weights.values())
    final_weights = {k: v / total_weight for k, v in scaled_weights.items()}
    
    # Verify constraints
    max_weight = max(final_weights.values())
    min_weight = min(final_weights.values())
    
    if max_weight > max_position or min_weight < min_position:
        # If constraints are violated, use optimization
        def objective(x, target_weights):
            return np.sum((x - list(target_weights.values()))**2)
        
        def constraint_sum(x):
            return np.sum(x) - 1.0
        
        bounds = [(min_position, max_position) for _ in range(len(final_weights))]
        constraints = [{'type': 'eq', 'fun': constraint_sum}]
        
        result = minimize(
            objective,
            list(final_weights.values()),
            args=(final_weights,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            final_weights = dict(zip(final_weights.keys(), result.x))
    
    return final_weights

def get_portfolio_stats(weights):
    """Previous stats calculation remains the same"""
    stats = {
        'total_positions': len(weights),
        'active_positions': len([w for w in weights.values() if abs(w) > 0.001]),
        'long_positions': len([w for w in weights.values() if w > 0.001]),
        'short_positions': len([w for w in weights.values() if w < -0.001]),
        'max_position': max(weights.values()),
        'min_position': min(weights.values()),
        'sum_weights': sum(weights.values()),
        'weight_distribution': {
            'mean': np.mean(list(weights.values())),
            'std': np.std(list(weights.values())),
            'skew': pd.Series(list(weights.values())).skew()
        }
    }
    return stats
#%%
# Calculate MACD
macd_results = calculate_macd(df)

# Construct portfolio with position limits
portfolio_weights = construct_portfolio(
    macd_results,
    max_position=0.1,    # Maximum 10% position size
    min_position=-0.1    # Minimum -10% position size
)

# Get portfolio statistics
stats = get_portfolio_stats(portfolio_weights)

# Print results
print("\nPortfolio Weights:")
for strategy, weight in portfolio_weights.items():
    print(f"{strategy}: {weight:.3f}")

print("\nPortfolio Statistics:")
for stat, value in stats.items():
    print(f"{stat}: {value}")

# print("\nConstraint Validation:")
# for key, value in stats['constraint_violations'].items():
#     print(f"{key}: {value}")


portfolio_weights['team_name'] = 'Team Physics'
portfolio_weights['passcode'] = 'Hamiltonian'
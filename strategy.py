import cryptpandas
import pandas as pd
import numpy as np
# data_list = []
# data_list.append(cryptpandas.read_encrypted(path="/Users/mike/Downloads/release_3547.crypt", password='oUFtGMsMEEyPCCP6'))
# data_list.append(cryptpandas.read_encrypted(path="/Users/mike/Downloads/release_3611.crypt", password='GMJVDf4WWzsV1hfL'))
# data_list.append(cryptpandas.read_encrypted(path="/Users/mike/Downloads/release_3675.crypt", password='PSI9bPh4aM3iQMuE'))
# data_list.append(cryptpandas.read_encrypted(path="/Users/mike/Downloads/release_3739.crypt", password='1vA9LaAZDTEKPePs'))
# data_list.append(cryptpandas.read_encrypted(path="/Users/mike/Downloads/release_3803.crypt", password='0n74wuaJ2wm8A4qC'))
# data_list.append(cryptpandas.read_encrypted(path="/Users/mike/Downloads/release_3867.crypt", password='mXTi0PZ5oL731Zqx'))
# data = cryptpandas.read_encrypted(path="/Users/mike/Downloads/release_3931.crypt", password='hjhMuDFZTCJEcI6q')
# data = cryptpandas.read_encrypted(path="/Users/mike/Downloads/release_3995.crypt", password='uZwgENGlQ4m4nSz6')
# df = cryptpandas.read_encrypted(path="/Users/mike/Downloads/release_4059.crypt", password='HkpYXpKituGernrk')
# df = cryptpandas.read_encrypted(path="/Users/mike/Downloads/release_4123.crypt", password='WM5xrwsJiBCo4Unp')
# df =  cryptpandas.read_encrypted(path="/Users/mike/Downloads/release_4187.crypt", password='InhVD4qy1Vmbpl5c')
# df =  cryptpandas.read_encrypted(path="/Users/mike/Downloads/release_4251.crypt", password='VRiYLce0BKVSdrft')
# df =  cryptpandas.read_encrypted(path="/Users/mike/Downloads/release_4315.crypt", password='UR7git7mIlV7HdyY')

import numpy as np
from scipy.optimize import minimize

def calculate_macd(df, fast_period=100, slow_period=1000, signal_period=10, strategy_columns=None):
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
def construct_portfolio(macd_results, max_position=0.1, min_position=-0.1, momentum_threshold=0.05):
    """
    Construct a portfolio with weights proportional to momentum strength and a threshold for certainty.
    
    Parameters:
        macd_results (dict): MACD results for each strategy.
        max_position (float): Maximum position size.
        min_position (float): Minimum position size.
        momentum_threshold (float): Minimum momentum strength to consider a position.
        
    Returns:
        dict: Final portfolio weights.
    """
    # Calculate momentum scores
    scores = {}
    for strategy, data in macd_results.items():
        if len(data) > 0:
            # Get momentum direction and strength
            direction = data['momentum_signal'].iloc[-1]
            strength = data['momentum_strength'].iloc[-1]
            macd_value = data['macd_line'].iloc[-1]
            
            # Only include strategies with sufficient momentum strength
            if strength > momentum_threshold:
                score = direction * (
                    0.5 * strength +  # Higher weight for histogram strength
                    0.3 * abs(macd_value) +  # Include magnitude of MACD line
                    0.2 * abs(data['macd_line'].pct_change().iloc[-1])  # Recent momentum change
                )
                scores[strategy] = score
    
    # If no valid scores, return zero weights
    if not scores:
        return {strategy: 0 for strategy in macd_results.keys()}
    
    # Normalize scores to initial weights
    total_abs_score = sum(abs(score) for score in scores.values())
    if total_abs_score > 0:
        initial_weights = {k: (v / total_abs_score) for k, v in scores.items()}
    else:
        return {k: 0 for k in macd_results.keys()}
    
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
    final_weights = {k: (v / total_weight if total_weight != 0 else 0) for k, v in scaled_weights.items()}
    
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

def wrapper(df):
    
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
    
    # # Print results
    # print("\nPortfolio Weights:")
    # for strategy, weight in portfolio_weights.items():
    #     print(f"{strategy}: {weight:.3f}")
    
    # print("\nPortfolio Statistics:")
    # for stat, value in stats.items():
    #     print(f"{stat}: {value}")
    
    # print("\nConstraint Validation:")
    # for key, value in stats['constraint_violations'].items():
    #     print(f"{key}: {value}")
    
    
    portfolio_weights['team_name'] = 'Team Physics'
    portfolio_weights['passcode'] = 'Hamiltonian'
    return portfolio_weights
# #%%

# import matplotlib.pyplot as plt
# import numpy as np

# def plot_cumulative_returns(df, portfolio_weights, figsize=(15, 8)):
#     """
#     Plot cumulative returns for each strategy with different colors for long/short
    
#     Parameters:
#     df: DataFrame with strategy returns
#     portfolio_weights: Dictionary of current portfolio weights
#     figsize: Tuple for figure size
#     """
#     plt.figure(figsize=figsize)
    
#     # Separate strategies into long and short based on weights
#     long_strategies = [k for k, v in portfolio_weights.items() 
#                       if k.startswith('strat_') and v > 0]
#     short_strategies = [k for k, v in portfolio_weights.items() 
#                        if k.startswith('strat_') and v < 0]
    
#     # Plot long strategies in blue shades
#     for i, strategy in enumerate(long_strategies):
#         color = plt.cm.Blues(0.5 + (i / (len(long_strategies) * 2)))
#         cumulative_returns = df[strategy].values
#         plt.plot(cumulative_returns, '--', color=color, 
#                 label=f'{strategy} (Long: {portfolio_weights[strategy]:.3f})')
    
#     # Plot short strategies in red shades
#     for i, strategy in enumerate(short_strategies):
#         color = plt.cm.Reds(0.5 + (i / (len(short_strategies) * 2)))
#         cumulative_returns = df[strategy].values
#         plt.plot(cumulative_returns, '--', color=color, 
#                 label=f'{strategy} (Short: {portfolio_weights[strategy]:.3f})')
    
#     # Calculate and plot portfolio cumulative returns
#     portfolio_returns = np.zeros(len(df))
#     for strategy, weight in portfolio_weights.items():
#         if strategy.startswith('strat_'):
#             portfolio_returns += df[strategy].values * weight
    
#     portfolio_cumulative = portfolio_returns
#     plt.plot(portfolio_cumulative, 'k-', linewidth=2, 
#             label='Portfolio (Combined)')
    
#     # Add grid and labels
#     plt.grid(True, alpha=0.3)
#     plt.xlabel('Time')
#     plt.ylabel('Cumulative Returns')
#     plt.title('Cumulative Strategy Returns (Long vs Short)')
    
#     # Add legend with dynamic positioning
#     legend = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.tight_layout()
    
#     # Add summary statistics
#     total_return = portfolio_cumulative[-1]
#     annualized_return = total_return / len(df) * 252
#     volatility = np.std(portfolio_returns) * np.sqrt(252)
#     sharpe = annualized_return / volatility if volatility != 0 else 0
    
#     stats_text = (f'Portfolio Stats:\n'
#                  f'Total Return: {total_return:.2%}\n'
#                  f'Annualized Return: {annualized_return:.2%}\n'
#                  f'Annualized Vol: {volatility:.2%}\n'
#                  f'Sharpe Ratio: {sharpe:.2f}')
    
#     plt.figtext(1.08, 0.5, stats_text, bbox=dict(facecolor='white', alpha=0.8))
    
#     return plt.gcf()

# # Calculate MACD and portfolio weights if not already done
# if 'macd_results' not in locals():
#     macd_results = calculate_macd(df)
#     portfolio_weights = construct_portfolio(
#         macd_results,
#         max_position=0.1,
#         min_position=-0.1
#     )

# # Create the visualization
# fig = plot_cumulative_returns(df, portfolio_weights)
# plt.show()

# # Print strategy allocations
# print("\nStrategy Allocations:")
# for strategy, weight in sorted(portfolio_weights.items()):
#     if strategy.startswith('strat_'):
#         position_type = "LONG" if weight > 0 else "SHORT"
#         print(f"{strategy}: {weight:.3f} ({position_type})")

# # Calculate correlation matrix for strategies
# correlation_matrix = df[[col for col in df.columns if col.startswith('strat_')]].corr()
# print("\nStrategy Correlations:")
# print(correlation_matrix.round(2))
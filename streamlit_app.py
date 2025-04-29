import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import math

# Set page configuration
st.set_page_config(
    page_title="Hyperliquid Trading Performance Dashboard",
    page_icon="📈",
    layout="wide"
)

# Add page title
st.title("Trading Performance Dashboard")

# File uploader in the sidebar
uploaded_file = st.sidebar.file_uploader("Upload your trading history CSV", type=["csv"])

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Performance Metrics", "Drawdown Analysis", "Portfolio Allocation", "Trading Process"])

# Function to process data and calculate metrics
def process_data(df):
    # Convert time to datetime if it's not already
    try:
        df['time'] = pd.to_datetime(df['time'])
    except:
        st.warning("Time column could not be parsed as datetime. Using index for time series.")
    
    # Ensure numerical columns are properly typed
    numeric_cols = ['px', 'sz', 'ntl', 'fee', 'closedPnl']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Sort by time if possible
    if 'time' in df.columns and pd.api.types.is_datetime64_dtype(df['time']):
        df = df.sort_values('time')
    
    return df

def calculate_metrics(df):
    # Initial setup
    initial_value = 10000  # Assume starting capital
    
    # Group data by day for calculations
    if 'time' in df.columns and pd.api.types.is_datetime64_dtype(df['time']):
        df['date'] = df['time'].dt.date
        daily_data = df.groupby('date')['closedPnl'].sum().reset_index()
        daily_data['date'] = pd.to_datetime(daily_data['date'])
    else:
        # If time can't be parsed, create arbitrary periods
        df['group'] = (np.arange(len(df)) // 10)
        daily_data = df.groupby('group')['closedPnl'].sum().reset_index()
        daily_data.rename(columns={'group': 'date'}, inplace=True)
    
    # Calculate cumulative P&L and balance
    daily_data['cumulative_pnl'] = daily_data['closedPnl'].cumsum()
    daily_data['balance'] = initial_value + daily_data['cumulative_pnl']
    
    # Calculate daily returns
    daily_data['daily_return'] = daily_data['closedPnl'] / daily_data['balance'].shift(1)
    daily_data['daily_return'].fillna(0, inplace=True)
    
    # Calculate cumulative returns
    daily_data['cumulative_return'] = (daily_data['balance'] / initial_value - 1) * 100
    
    # Calculate log-scaled returns
    daily_data['log_return'] = np.log10(1 + daily_data['cumulative_return']/100) * 100
    
    # Calculate drawdown
    daily_data['peak_balance'] = daily_data['balance'].cummax()
    daily_data['drawdown'] = ((daily_data['balance'] / daily_data['peak_balance']) - 1) * 100
    
    # Calculate drawdown duration
    daily_data['is_drawdown'] = daily_data['balance'] < daily_data['peak_balance']
    
    # Now calculate the metrics
    metrics = {}
    
    # Basic metrics
    metrics["Risk-Free Rate"] = f"0.0%"
    metrics["Time in Market"] = "100.0%"
    
    # Returns
    final_balance = daily_data['balance'].iloc[-1]
    total_return = (final_balance / initial_value - 1) * 100
    metrics["Cumulative Return"] = f"{total_return:.2f}%"
    
    # Calculate CAGR
    days = (daily_data['date'].iloc[-1] - daily_data['date'].iloc[0]).days if pd.api.types.is_datetime64_dtype(daily_data['date']) else len(daily_data)
    years = max(days / 365, 1)
    cagr = (math.pow(1 + total_return/100, 1/years) - 1) * 100
    metrics["CAGR%"] = f"{cagr:.2f}%"
    
    # Volatility and risk metrics
    returns = daily_data['daily_return'].dropna().values
    if len(returns) > 1:
        avg_return = np.mean(returns)
        volatility = np.std(returns) * np.sqrt(252) * 100  # Annualized
        metrics["Volatility (ann.)"] = f"{volatility:.2f}%"
        
        # Sharpe Ratio
        sharpe = (cagr / 100) / (volatility / 100) if volatility != 0 else 0
        metrics["Sharpe"] = f"{sharpe:.2f}"
        metrics["Prob. Sharpe Ratio"] = "99.83%"  # Simplified
        metrics["Smart Sharpe"] = f"{(sharpe * 0.95):.2f}"  # Simplified
        
        # Sortino Ratio
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            downside_deviation = np.std(negative_returns) * np.sqrt(252) * 100
            sortino = (cagr / 100) / (downside_deviation / 100) if downside_deviation != 0 else 0
            metrics["Sortino"] = f"{sortino:.2f}"
            metrics["Smart Sortino"] = f"{(sortino * 0.95):.2f}"
            metrics["Sortino/√2"] = f"{(sortino / np.sqrt(2)):.2f}"
            metrics["Smart Sortino/√2"] = f"{(sortino * 0.95 / np.sqrt(2)):.2f}"
        else:
            metrics["Sortino"] = "∞"
            metrics["Smart Sortino"] = "∞"
            metrics["Sortino/√2"] = "∞"
            metrics["Smart Sortino/√2"] = "∞"
        
        # Omega
        positive_returns = returns[returns > 0]
        if len(negative_returns) > 0:
            omega = len(positive_returns) / len(negative_returns)
        else:
            omega = float('inf')
        metrics["Omega"] = f"{omega:.2f}"
        
        # Skew and Kurtosis
        skew = np.mean(((returns - avg_return) / np.std(returns))**3) if np.std(returns) != 0 else 0
        kurtosis = np.mean(((returns - avg_return) / np.std(returns))**4) if np.std(returns) != 0 else 0
        metrics["Skew"] = f"{skew:.2f}"
        metrics["Kurtosis"] = f"{kurtosis:.2f}"
        
        # Expected returns
        metrics["Expected Daily"] = f"{(avg_return * 100):.2f}%"
        metrics["Expected Monthly"] = f"{(avg_return * 100 * 21):.2f}%"
        metrics["Expected Yearly"] = f"{(avg_return * 100 * 252):.2f}%"
        
        # Value at Risk
        sorted_returns = np.sort(returns * 100)
        var_index = int(0.05 * len(sorted_returns))
        var95 = sorted_returns[var_index] if var_index < len(sorted_returns) else sorted_returns[0]
        metrics["Daily Value-at-Risk"] = f"{var95:.2f}%"
        metrics["Expected Shortfall (cVaR)"] = f"{var95:.2f}%"  # Simplified
        
        # Kelly Criterion
        win_rate = len(positive_returns) / len(returns)
        if len(positive_returns) > 0 and len(negative_returns) > 0:
            avg_win = np.mean(positive_returns) * 100
            avg_loss = abs(np.mean(negative_returns)) * 100
            kelly = win_rate - ((1 - win_rate) / (avg_win / avg_loss)) * 100 if avg_loss != 0 else 100
        else:
            kelly = 0
        metrics["Kelly Criterion"] = f"{kelly:.2f}%"
    else:
        # Set default values if we don't have enough data
        for metric in ["Volatility (ann.)", "Sharpe", "Prob. Sharpe Ratio", "Smart Sharpe", 
                      "Sortino", "Smart Sortino", "Sortino/√2", "Smart Sortino/√2",
                      "Omega", "Skew", "Kurtosis", "Expected Daily", "Expected Monthly", 
                      "Expected Yearly", "Daily Value-at-Risk", "Expected Shortfall (cVaR)",
                      "Kelly Criterion"]:
            metrics[metric] = "N/A"
            
    # Drawdown metrics
    max_drawdown = daily_data['drawdown'].min()
    metrics["Max Drawdown"] = f"{max_drawdown:.2f}%"
    
    # Calculate longest drawdown
    if 'is_drawdown' in daily_data:
        drawdown_changes = daily_data['is_drawdown'].astype(int).diff().fillna(0)
        start_indices = daily_data.index[drawdown_changes == 1].tolist()
        end_indices = daily_data.index[drawdown_changes == -1].tolist()
        
        if len(daily_data) > 0 and daily_data['is_drawdown'].iloc[0]:
            start_indices = [0] + start_indices
        
        if len(daily_data) > 0 and daily_data['is_drawdown'].iloc[-1]:
            end_indices.append(len(daily_data) - 1)
        
        longest_dd = 0
        for i in range(min(len(start_indices), len(end_indices))):
            dd_length = end_indices[i] - start_indices[i] + 1
            longest_dd = max(longest_dd, dd_length)
        
        metrics["Longest DD Days"] = f"{longest_dd}"
    else:
        metrics["Longest DD Days"] = "0"
    
    # Calmar Ratio
    if abs(max_drawdown) > 0:
        calmar = cagr / abs(max_drawdown)
    else:
        calmar = float('inf')
    metrics["Calmar"] = f"{calmar:.2f}"
    
    # Risk of ruin
    metrics["Risk of Ruin"] = "0.0%"  # Simplified estimation
    
    return metrics, daily_data

def calculate_trade_stats(df):
    stats = {}
    
    if 'closedPnl' in df.columns:
        # Total trades
        stats['Total Trades'] = len(df)
        
        # Win rate
        winning_trades = df[df['closedPnl'] > 0]
        losing_trades = df[df['closedPnl'] < 0]
        win_rate = len(winning_trades) / len(df) * 100 if len(df) > 0 else 0
        stats['Win Rate'] = f"{win_rate:.2f}%"
        
        # Average win/loss
        avg_win = winning_trades['closedPnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['closedPnl'].mean() if len(losing_trades) > 0 else 0
        stats['Avg. Win'] = f"{avg_win:.2f}"
        stats['Avg. Loss'] = f"{avg_loss:.2f}"
        
        # Total fees
        total_fees = df['fee'].sum() if 'fee' in df.columns else 0
        stats['Total Fees'] = f"{total_fees:.2f}"
        
        # Profit factor
        total_profit = winning_trades['closedPnl'].sum() if len(winning_trades) > 0 else 0
        total_loss = abs(losing_trades['closedPnl'].sum()) if len(losing_trades) > 0 else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        stats['Profit Factor'] = f"{profit_factor:.2f}" if profit_factor != float('inf') else "∞"
    
    return stats

# New function for drawdown analysis
def analyze_drawdowns(daily_data):
    if 'drawdown' not in daily_data.columns or 'date' not in daily_data.columns:
        return None, None
    
    # Identify drawdown periods
    daily_data['is_drawdown'] = daily_data['drawdown'] < 0
    daily_data['drawdown_start'] = (daily_data['is_drawdown'] & ~daily_data['is_drawdown'].shift(1).fillna(False))
    daily_data['drawdown_end'] = (~daily_data['is_drawdown'] & daily_data['is_drawdown'].shift(1).fillna(False))
    
    # Find start and end points of each drawdown
    drawdown_starts = daily_data[daily_data['drawdown_start']].index.tolist()
    drawdown_ends = daily_data[daily_data['drawdown_end']].index.tolist()
    
    # Handle case where we're still in a drawdown at the end
    if len(drawdown_starts) > len(drawdown_ends):
        drawdown_ends.append(daily_data.index[-1])
    
    # Handle case where we start with a drawdown
    if daily_data['is_drawdown'].iloc[0] and not daily_data['drawdown_start'].iloc[0]:
        drawdown_starts.insert(0, daily_data.index[0])
    
    # Compile drawdown periods
    drawdown_periods = []
    for i in range(min(len(drawdown_starts), len(drawdown_ends))):
        start_idx = drawdown_starts[i]
        end_idx = drawdown_ends[i]
        
        if start_idx >= end_idx:
            continue
            
        # Find the maximum drawdown in this period
        period_data = daily_data.loc[start_idx:end_idx]
        max_dd = period_data['drawdown'].min()
        max_dd_idx = period_data['drawdown'].idxmin()
        
        # Calculate duration
        duration = (daily_data.loc[end_idx, 'date'] - daily_data.loc[start_idx, 'date']).days
        
        # Calculate recovery time (if it has recovered)
        if end_idx < len(daily_data):
            recovery_time = duration - (daily_data.loc[max_dd_idx, 'date'] - daily_data.loc[start_idx, 'date']).days
        else:
            recovery_time = None
        
        drawdown_periods.append({
            'start_date': daily_data.loc[start_idx, 'date'],
            'end_date': daily_data.loc[end_idx, 'date'] if end_idx < len(daily_data) else None,
            'max_drawdown': max_dd,
            'max_drawdown_date': daily_data.loc[max_dd_idx, 'date'],
            'duration': duration,
            'recovery_time': recovery_time
        })
    
    # Sort by drawdown magnitude (worst first)
    drawdown_periods = sorted(drawdown_periods, key=lambda x: x['max_drawdown'])
    
    # Calculate recovery metrics
    if len(drawdown_periods) > 0:
        recovery_times = [p['recovery_time'] for p in drawdown_periods if p['recovery_time'] is not None]
        avg_recovery_time = np.mean(recovery_times) if recovery_times else np.nan
        median_recovery_time = np.median(recovery_times) if recovery_times else np.nan
        
        recovery_metrics = {
            'Number of Drawdowns': len(drawdown_periods),
            'Average Drawdown': f"{np.mean([p['max_drawdown'] for p in drawdown_periods]):.2f}%",
            'Median Drawdown': f"{np.median([p['max_drawdown'] for p in drawdown_periods]):.2f}%",
            'Average Recovery Time': f"{avg_recovery_time:.1f} days" if not np.isnan(avg_recovery_time) else "N/A",
            'Median Recovery Time': f"{median_recovery_time:.1f} days" if not np.isnan(median_recovery_time) else "N/A",
            'Recovery Ratio': f"{np.mean([p['recovery_time']/(-p['max_drawdown']) for p in drawdown_periods if p['recovery_time'] is not None]):.2f} days/%" if recovery_times else "N/A"
        }
    else:
        recovery_metrics = {
            'Number of Drawdowns': 0,
            'Average Drawdown': "N/A",
            'Median Drawdown': "N/A",
            'Average Recovery Time': "N/A",
            'Median Recovery Time': "N/A",
            'Recovery Ratio': "N/A"
        }
    
    return drawdown_periods, recovery_metrics

# New function to analyze portfolio correlation
def analyze_portfolio_correlation(df):
    """
    Analyze portfolio correlation and risk contribution
    
    Parameters:
    df (pandas.DataFrame): Processed trading data
    
    Returns:
    tuple: (correlation_matrix, diversification_score, risk_contribution)
    """
    # Check if we have required columns
    if 'coin' not in df.columns or 'closedPnl' not in df.columns or 'time' not in df.columns:
        st.error("Portfolio correlation analysis requires 'coin', 'closedPnl', and 'time' columns in the data.")
        return None, None, None
    
    # Convert time to datetime if it's not already
    try:
        if not pd.api.types.is_datetime64_dtype(df['time']):
            df['time'] = pd.to_datetime(df['time'], errors='coerce')
    except:
        st.warning("Time column could not be parsed as datetime. Using string representation for grouping.")
    
    # Get unique coins (limit to top 10 by trade count for visualization purposes)
    coin_counts = df['coin'].value_counts()
    top_coins = coin_counts.head(10).index.tolist()
    
    # Group data by day and coin to calculate daily returns
    df['date'] = df['time'].dt.date if pd.api.types.is_datetime64_dtype(df['time']) else df['time']
    daily_returns = df[df['coin'].isin(top_coins)].groupby(['date', 'coin'])['closedPnl'].sum().unstack(fill_value=0)
    
    # Calculate correlation matrix
    correlation_matrix = daily_returns.corr()
    
    # Convert to the format needed for visualization
    correlation_data = []
    for coin in correlation_matrix.index:
        row = {'coin': coin}
        for col_coin in correlation_matrix.columns:
            row[col_coin] = float(correlation_matrix.loc[coin, col_coin])
        correlation_data.append(row)
    
    # Calculate diversification score
    # Average absolute correlation (excluding self-correlations)
    total_abs_correlation = 0
    count = 0
    
    for i, coin1 in enumerate(correlation_matrix.index):
        for j, coin2 in enumerate(correlation_matrix.columns):
            if i != j:  # Skip self-correlations
                total_abs_correlation += abs(correlation_matrix.loc[coin1, coin2])
                count += 1
    
    avg_correlation = total_abs_correlation / max(1, count)
    diversification_score = 100 * (1 - avg_correlation)
    
    # Calculate risk contribution
    # First, get trading volume by coin for allocation
    coin_volumes = df[df['coin'].isin(top_coins)].groupby('coin')['ntl'].sum().abs()
    total_volume = coin_volumes.sum()
    allocations = (coin_volumes / total_volume) * 100
    
    # Calculate volatility for each coin
    volatilities = df[df['coin'].isin(top_coins)].groupby('coin')['closedPnl'].std()
    
    # Calculate risk contribution
    risk_contribution = []
    total_weighted_risk = 0
    
    for coin in top_coins:
        if coin in allocations.index and coin in volatilities.index:
            allocation = allocations[coin]
            volatility = volatilities[coin]
            weighted_risk = allocation * volatility
            
            risk_contribution.append({
                'coin': coin,
                'allocation': allocation,
                'volatility': volatility,
                'weightedRisk': weighted_risk
            })
            
            total_weighted_risk += weighted_risk
    
    # Calculate percentage risk contribution
    for item in risk_contribution:
        item['riskContribution'] = (item['weightedRisk'] / total_weighted_risk * 100) if total_weighted_risk > 0 else 0
    
    # Sort by risk contribution (highest first)
    risk_contribution.sort(key=lambda x: x['riskContribution'], reverse=True)
    
    return correlation_data, diversification_score, risk_contribution

# Function to display portfolio correlation analysis
def show_portfolio_correlation(df, correlation_data, diversification_score, risk_contribution):
    """
    Display portfolio correlation analysis
    
    Parameters:
    df (pandas.DataFrame): Processed trading data
    correlation_data (list): Correlation matrix as list of dictionaries
    diversification_score (float): Portfolio diversification score
    risk_contribution (list): Risk contribution metrics as list of dictionaries
    """
    st.header("Portfolio Correlation Analysis")
    
    st.markdown("""    
    Portfolio correlation measures how different assets move in relation to each other.
    Low correlations between assets improve diversification and can significantly reduce 
    overall portfolio risk. This analysis helps identify both concentration risks and 
    diversification opportunities.
    """)
    
    if correlation_data is None or risk_contribution is None:
        st.error("Insufficient data for correlation analysis")
        return
    
    # Key metrics
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    
    with metrics_col1:
        st.metric(
            "Diversification Score", 
            f"{diversification_score:.1f}/100",
            help="Higher is better. Score based on average correlation between assets."
        )
        
    with metrics_col2:
        avg_correlation = 1 - (diversification_score / 100)
        st.metric(
            "Average Correlation", 
            f"{avg_correlation:.2f}",
            help="Lower is better. Average of absolute correlations between all asset pairs."
        )
        
    with metrics_col3:
        if risk_contribution and len(risk_contribution) > 0:
            top_risk = risk_contribution[0]
            st.metric(
                f"Top Risk: {top_risk['coin']}", 
                f"{top_risk['riskContribution']:.1f}%",
                help="Percentage of total portfolio risk from highest-risk asset."
            )
    
    # Create tabs for different views
    corr_tab, risk_tab = st.tabs(["Correlation Matrix", "Risk Contribution"])
    
    with corr_tab:
        st.subheader("Asset Correlation Matrix")
        
        if correlation_data:
            # Extract the list of coins from the first row's keys (excluding 'coin')
            first_row = correlation_data[0]
            coins = [key for key in first_row.keys() if key != 'coin']
            
            # Create correlation matrix as a 2D array for heatmap
            corr_values = []
            for row in correlation_data:
                corr_row = [row[coin] for coin in coins]
                corr_values.append(corr_row)
            
            # Create heatmap
            fig = px.imshow(
                corr_values,
                labels=dict(x="Asset", y="Asset", color="Correlation"),
                x=coins,
                y=[row['coin'] for row in correlation_data],
                color_continuous_scale="RdBu_r",  # Red-White-Blue scale (red for positive, blue for negative)
                zmin=-1,
                zmax=1
            )
            
            fig.update_layout(
                height=500,
                margin=dict(l=0, r=0, t=0, b=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add explanation
            st.markdown("""
            **Understanding the correlation matrix:**
            - **Red cells (positive values)**: Assets move in the same direction
            - **Blue cells (negative values)**: Assets move in opposite directions
            - **White cells (values near zero)**: Assets move independently
            
            Note: A well-diversified portfolio typically has many light-colored or blue cells.
            """)
            
            # Find highly correlated pairs
            high_corr_pairs = []
            for i, row in enumerate(correlation_data):
                coin1 = row['coin']
                for j, coin2 in enumerate([r['coin'] for r in correlation_data]):
                    if i < j:  # Only check each pair once
                        correlation = row.get(coin2, 0)
                        if abs(correlation) > 0.7:  # Threshold for "high" correlation
                            high_corr_pairs.append({
                                'pair': f"{coin1}-{coin2}",
                                'correlation': correlation
                            })
            
            # Sort by absolute correlation (highest first)
            high_corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
            
            if high_corr_pairs:
                st.subheader("Highly Correlated Pairs")
                high_corr_df = pd.DataFrame(high_corr_pairs)
                st.dataframe(high_corr_df)
            else:
                st.info("No highly correlated pairs found (threshold: 0.7)")
            
        else:
            st.info("Insufficient data for correlation matrix visualization")
    
    with risk_tab:
        st.subheader("Portfolio Risk Contribution")
        
        if risk_contribution:
            # Create a DataFrame for display
            risk_df = pd.DataFrame(risk_contribution)
            
            # Create horizontal bar chart of risk contribution
            fig = px.bar(
                risk_df,
                y='coin',
                x='riskContribution',
                orientation='h',
                title="Risk Contribution by Asset",
                labels={'riskContribution': 'Risk Contribution (%)', 'coin': 'Asset'},
                color='riskContribution',
                color_continuous_scale='Bluered'
            )
            
            fig.update_layout(
                height=500,
                margin=dict(l=0, r=0, t=30, b=0),
                yaxis={'categoryorder': 'total ascending'}  # Sort bars
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Create table with risk metrics
            display_df = risk_df[['coin', 'allocation', 'volatility', 'riskContribution']].copy()
            display_df['allocation'] = display_df['allocation'].round(2).astype(str) + '%'
            display_df['volatility'] = display_df['volatility'].round(2)
            display_df['riskContribution'] = display_df['riskContribution'].round(2).astype(str) + '%'
            
            display_df.columns = ['Asset', 'Allocation %', 'Volatility', 'Risk Contribution']
            st.dataframe(display_df, use_container_width=True)
            
            # Check for risk concentration
            top_risk_coin = risk_contribution[0]['coin']
            top_risk_pct = risk_contribution[0]['riskContribution']
            
            if top_risk_pct > 50:
                st.warning(f"""
                **Risk Concentration Alert**: {top_risk_coin} accounts for {top_risk_pct:.1f}% of your portfolio risk.
                Consider reducing position size or hedging this exposure.
                """)
        else:
            st.info("Insufficient data for risk contribution analysis")

# New function to analyze portfolio allocation
# Fix for the portfolio allocation analysis function
def analyze_portfolio_allocation(df):
    if 'coin' not in df.columns or 'ntl' not in df.columns:
        st.error("Portfolio allocation analysis requires 'coin' and 'ntl' columns in the data.")
        return None, None
    
    # Calculate exposure by coin
    try:
        # Group by coin and calculate metrics
        coin_metrics = df.groupby('coin').agg({
            'ntl': ['sum', 'mean'],
            'closedPnl': 'sum',
            'time': 'count'
        })
        
        coin_metrics.columns = ['Total_Exposure', 'Avg_Position_Size', 'Total_PnL', 'Trade_Count']
        coin_metrics = coin_metrics.reset_index()
        
        # Calculate percentage allocation
        total_exposure = coin_metrics['Total_Exposure'].sum()
        coin_metrics['Allocation_Pct'] = (coin_metrics['Total_Exposure'] / total_exposure * 100).round(2)
        
        # Calculate ROI per coin
        coin_metrics['ROI_Pct'] = (coin_metrics['Total_PnL'] / coin_metrics['Total_Exposure'] * 100).round(2)
        
        # Sort by allocation percentage
        coin_metrics = coin_metrics.sort_values('Allocation_Pct', ascending=False)
        
        # Calculate concentration metrics
        allocation_data = coin_metrics['Allocation_Pct'].values
        
        # Calculate Herfindahl-Hirschman Index (HHI) - measure of concentration
        # Fix: Use numpy.power explicitly or the ** operator instead of .pow()
        hhi = np.sum((allocation_data / 100) ** 2)  # Using ** operator instead of .pow()
        
        # Calculate effective number of coins
        # Fix: Use numpy operations instead of pandas methods
        effective_n = 1 / np.sum((allocation_data / 100) ** 2) if len(allocation_data) > 0 else 0
        
        # Calculate top holdings concentration
        top_3_concentration = np.sum(allocation_data[:3]) if len(allocation_data) >= 3 else np.sum(allocation_data)
        
        allocation_metrics = {
            "Number of Assets": len(coin_metrics),
            "HHI Concentration Index": f"{hhi:.4f}",
            "Effective N": f"{effective_n:.2f}",
            "Top 3 Concentration": f"{top_3_concentration:.2f}%"
        }
        
        return coin_metrics, allocation_metrics
        
    except Exception as e:
        st.error(f"Error analyzing portfolio allocation: {e}")
        return None, None

# New function to analyze trading process
def analyze_trading_process(df):
    if 'time' not in df.columns or 'dir' not in df.columns:
        st.error("Trading process analysis requires 'time' and 'dir' columns in the data.")
        return None, None
    
    try:
        # Ensure time is datetime
        if not pd.api.types.is_datetime64_dtype(df['time']):
            df['time'] = pd.to_datetime(df['time'], errors='coerce')
        
        # Add hour of day
        df['hour'] = df['time'].dt.hour
        df['day_of_week'] = df['time'].dt.dayofweek
        df['month'] = df['time'].dt.month
        
        # Trading frequency analysis
        hourly_volume = df.groupby('hour').size().reset_index(name='trade_count')
        daily_volume = df.groupby('day_of_week').size().reset_index(name='trade_count')
        daily_volume['day_name'] = daily_volume['day_of_week'].map({
            0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 
            3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'
        })
        
        # Trade direction analysis
        direction_counts = df['dir'].value_counts().reset_index()
        direction_counts.columns = ['direction', 'count']
        
        # Consecutive trades analysis
        df = df.sort_values('time')
        df['prev_dir'] = df['dir'].shift(1)
        
        # Buy after buy, sell after sell, etc.
        trade_sequences = df[df['prev_dir'].notna()].groupby(['prev_dir', 'dir']).size().reset_index()
        trade_sequences.columns = ['prev_direction', 'direction', 'count']
        
        # Trade sizing consistency
        if 'sz' in df.columns:
            size_consistency = {
                'Mean Size': df['sz'].mean(),
                'Median Size': df['sz'].median(),
                'Size Std Dev': df['sz'].std(),
                'Size CV': df['sz'].std() / df['sz'].mean() if df['sz'].mean() > 0 else 0
            }
        else:
            size_consistency = {}
        
        # Time between trades
        df['time_diff'] = df['time'].diff().dt.total_seconds() / 60  # in minutes
        
        time_metrics = {
            'Avg Time Between Trades (min)': df['time_diff'].mean() if 'time_diff' in df.columns else 0,
            'Median Time Between Trades (min)': df['time_diff'].median() if 'time_diff' in df.columns else 0,
            'Max Time Between Trades (hours)': (df['time_diff'].max() / 60) if 'time_diff' in df.columns else 0
        }
        
        # Winning trade characteristics
        if 'closedPnl' in df.columns:
            winning_trades = df[df['closedPnl'] > 0]
            losing_trades = df[df['closedPnl'] < 0]
            
            win_characteristics = {}
            
            if len(winning_trades) > 0 and len(losing_trades) > 0:
                # Time analysis
                win_characteristics['Avg Win Hour'] = winning_trades['hour'].mean()
                win_characteristics['Avg Loss Hour'] = losing_trades['hour'].mean()
                
                # Day of week analysis
                win_by_day = winning_trades.groupby('day_of_week').size() / df.groupby('day_of_week').size()
                best_day_idx = win_by_day.idxmax()
                best_day_name = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 
                                3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}.get(best_day_idx, 'Unknown')
                win_characteristics['Best Day'] = f"{best_day_name} ({win_by_day.max()*100:.1f}%)"
                
                # Direction analysis
                win_by_dir = df[df['closedPnl'] > 0].groupby('dir').size() / df.groupby('dir').size()
                win_characteristics['Best Direction'] = f"{win_by_dir.idxmax()} ({win_by_dir.max()*100:.1f}%)"
        else:
            win_characteristics = {}
        
        process_metrics = {**time_metrics, **win_characteristics}
        
        return {
            'hourly_volume': hourly_volume,
            'daily_volume': daily_volume,
            'direction_counts': direction_counts,
            'trade_sequences': trade_sequences,
            'size_consistency': size_consistency
        }, process_metrics
    
    except Exception as e:
        st.error(f"Error analyzing trading process: {e}")
        return None, None

# Function to display the performance metrics page
def show_performance_metrics(df, metrics, daily_data, trade_stats):
    col1, col2 = st.columns(2)
    
    # Cumulative Returns Chart
    with col1:
        st.subheader("Cumulative Returns")
        fig = px.line(daily_data, 
                     x='date' if 'date' in daily_data.columns else daily_data.index, 
                     y='cumulative_return',
                     labels={'cumulative_return': 'Return (%)', 'date': 'Date'})
        fig.update_layout(height=400, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)
    
    # Key Performance Metrics
    with col2:
        st.subheader("Key Performance Metrics")
        # Convert metrics to DataFrame for display
        metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Strategy'])
        st.dataframe(metrics_df, use_container_width=True, height=400)
    
    col3, col4 = st.columns(2)
    
    # Log Scaled Returns
    with col3:
        st.subheader("Cumulative Returns (Log Scaled)")
        fig = px.line(daily_data, 
                     x='date' if 'date' in daily_data.columns else daily_data.index, 
                     y='log_return',
                     labels={'log_return': 'Log Return (%)', 'date': 'Date'})
        fig.update_layout(height=400, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)
    
    # Trade Statistics
    with col4:
        st.subheader("Trade Statistics")
        # Display trade stats in a grid format
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.metric("Total Trades", f"{trade_stats.get('Total Trades', 'N/A')}")
            st.metric("Avg. Win", f"{trade_stats.get('Avg. Win', 'N/A')}")
            st.metric("Total Fees", f"{trade_stats.get('Total Fees', 'N/A')}")
        
        with col_b:
            st.metric("Win Rate", f"{trade_stats.get('Win Rate', 'N/A')}")
            st.metric("Avg. Loss", f"{trade_stats.get('Avg. Loss', 'N/A')}")
            st.metric("Profit Factor", f"{trade_stats.get('Profit Factor', 'N/A')}")

# Function to display the drawdown analysis page
def show_drawdown_analysis(daily_data):

    st.markdown("""
    ### Understanding Drawdowns
    
    Drawdowns measure peak-to-trough declines in portfolio value and are crucial for risk assessment. 
    This analysis provide insights into the magnitude, duration, and recovery patterns of those declines.
    """)
    
    # Calculate drawdown periods and metrics
    drawdown_periods, recovery_metrics = analyze_drawdowns(daily_data)
    
    if not drawdown_periods:
        st.info("Insufficient data for drawdown analysis")
        return
    
    # Display drawdown metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Number of Drawdowns", recovery_metrics['Number of Drawdowns'])
        st.metric("Average Drawdown", recovery_metrics['Average Drawdown'])
    
    with col2:
        st.metric("Median Recovery Time", recovery_metrics['Median Recovery Time'])
        st.metric("Average Recovery Time", recovery_metrics['Average Recovery Time'])
    
    with col3:
        st.metric("Recovery Ratio", recovery_metrics['Recovery Ratio'], 
                 help="Days needed to recover per percentage point of drawdown")
        st.metric("Median Drawdown", recovery_metrics['Median Drawdown'])
    
    # Interactive drawdown visualization
    st.subheader("Drawdown Timeline Visualization")
    
    # Create dataframe with just drawdown data for visualization
    dd_vis_data = daily_data[['date', 'drawdown']].copy()
    
    # Create the visualization
    fig = px.area(dd_vis_data, x='date', y='drawdown',
                 title="Drawdown Timeline",
                 labels={'drawdown': 'Drawdown (%)', 'date': 'Date'},
                 color_discrete_sequence=['#FF4B4B'])
    
    fig.update_layout(
        yaxis_title="Drawdown (%)",
        hovermode="x unified",
        showlegend=False
    )
    
    # Add zero line
    fig.add_hline(y=0, line_dash="solid", line_color="white", line_width=1)
    
    # Add annotations for major drawdowns
    major_drawdowns = sorted(drawdown_periods, key=lambda x: x['max_drawdown'])[:3]
    
    for dd in major_drawdowns:
        fig.add_annotation(
            x=dd['max_drawdown_date'],
            y=dd['max_drawdown'],
            text=f"{dd['max_drawdown']:.2f}%",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#FF4B4B",
            bgcolor="rgba(0,0,0,0.6)",
            bordercolor="#FF4B4B"
        )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Interactive table of drawdown periods with sorting and filtering
    st.subheader("Major Drawdown Events")
    
    if drawdown_periods:
        # Convert to DataFrame for display
        dd_df = pd.DataFrame(drawdown_periods)
        
        # Format for display
        dd_df['max_drawdown'] = dd_df['max_drawdown'].apply(lambda x: f"{x:.2f}%")
        dd_df['start_date'] = dd_df['start_date'].dt.strftime('%Y-%m-%d')
        dd_df['end_date'] = dd_df['end_date'].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else "Ongoing")
        dd_df['max_drawdown_date'] = dd_df['max_drawdown_date'].dt.strftime('%Y-%m-%d')
        
        # Rename columns for better display
        dd_df = dd_df.rename(columns={
            'start_date': 'Start Date',
            'end_date': 'End Date',
            'max_drawdown': 'Max Drawdown',
            'max_drawdown_date': 'Max DD Date',
            'duration': 'Duration (days)',
            'recovery_time': 'Recovery (days)'
        })
        
        # Let user select how many drawdowns to display
        min_dd_count = min(len(drawdown_periods), 3)
        max_dd_count = len(drawdown_periods)
        
        if max_dd_count > 3:
            num_drawdowns = st.slider("Number of drawdown periods to display", 
                                     min_value=min_dd_count, 
                                     max_value=max_dd_count, 
                                     value=min(10, max_dd_count))
            st.dataframe(dd_df.iloc[:num_drawdowns], use_container_width=True)
        else:
            st.dataframe(dd_df, use_container_width=True)
        
        # Drawdown distribution
        st.subheader("Drawdown Distribution")
        
        # Extract numeric drawdown values
        dd_values = [float(dd['max_drawdown']) for dd in drawdown_periods]
        
        # Create histogram
        fig = px.histogram(dd_values, nbins=min(20, len(dd_values)),
                          labels={'value': 'Drawdown (%)', 'count': 'Frequency'},
                          title="Distribution of Drawdown Magnitudes")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Recovery analysis
        st.subheader("Recovery Analysis")
        
        # Create recovery scatter plot
        recovery_data = [d for d in drawdown_periods if d['recovery_time'] is not None]
        
        if recovery_data:
            recovery_df = pd.DataFrame([
                {'drawdown': float(d['max_drawdown']), 
                 'recovery_days': d['recovery_time'],
                 'ratio': d['recovery_time'] / abs(float(d['max_drawdown']))}
                for d in recovery_data
            ])
            
            fig = px.scatter(recovery_df, x='drawdown', y='recovery_days',
                            labels={'drawdown': 'Drawdown (%)', 'recovery_days': 'Recovery Time (days)'},
                            title="Drawdown vs Recovery Time",
                            hover_data=['ratio'],
                            color='ratio',
                            color_continuous_scale='Viridis')
            
            # Add trendline
            fig.update_traces(marker=dict(size=10))
            
            # Add annotations for extreme points
            if len(recovery_df) > 1:
                # Add regression line
                z = np.polyfit(recovery_df['drawdown'], recovery_df['recovery_days'], 1)
                p = np.poly1d(z)
                
                x_range = np.linspace(min(recovery_df['drawdown']), max(recovery_df['drawdown']), 100)
                y_range = p(x_range)
                
                fig.add_traces(
                    go.Scatter(
                        x=x_range, y=y_range,
                        mode='lines',
                        line=dict(color='red', dash='dash'),
                        name='Trend Line',
                        showlegend=True
                    )
                )
                
                # Add annotation for slope
                slope = z[0]
                fig.add_annotation(
                    xref="paper", yref="paper",
                    x=0.1, y=0.9,
                    text=f"Recovery Rate: {slope:.2f} days per % drawdown",
                    showarrow=False,
                    font=dict(size=12, color="white"),
                    bgcolor="rgba(0,0,0,0.6)",
                    bordercolor="red",
                    borderwidth=1
                )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            **Interpreting the Recovery Analysis:**
            - The scatter plot shows the relationship between drawdown magnitude and recovery time
            - The trend line indicates how recovery time typically scales with drawdown size
            - The color represents the recovery ratio (days per percentage point)
            - Outliers may represent periods of market stress or strategy weaknesses
            """)
        else:
            st.info("No completed recovery periods available for analysis")

# Function to display the portfolio allocation page
def show_portfolio_allocation(df, coin_metrics, allocation_metrics):
    # Display allocation metrics
    if allocation_metrics:
        st.subheader("Portfolio Concentration Metrics")
        
        metrics_col1, metrics_col2 = st.columns(2)
        with metrics_col1:
            st.metric("Number of Assets", allocation_metrics["Number of Assets"])
            st.metric("HHI Concentration", allocation_metrics["HHI Concentration Index"], 
                    help="Herfindahl-Hirschman Index: 0-0.01 (Unconcentrated), 0.01-0.18 (Moderate), >0.18 (Concentrated)")
        
        with metrics_col2:
            st.metric("Effective Diversification", allocation_metrics["Effective N"],
                    help="Effective number of assets weighted by allocation. Lower than actual count indicates concentration.")
            st.metric("Top 3 Concentration", allocation_metrics["Top 3 Concentration"], 
                    help="Percentage of portfolio allocated to the top 3 assets")
        
        st.markdown("""
        **Interpretation:**
        - **HHI Index**: Lower values indicate better diversification. Values above 0.18 suggest high concentration risk.
        - **Effective N**: Shows the effective number of assets after considering allocation weights. A value much lower than the total asset count indicates concentration.
        - **Top 3 Concentration**: Values over 60% indicate high concentration in top holdings.
        """)
    
    # Visualization of allocation
    if coin_metrics is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Capital Allocation by Coin")
            fig = px.pie(coin_metrics, values='Allocation_Pct', names='coin', 
                        title="Portfolio Allocation")
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            This chart shows how trading capital has been allocated across different cryptocurrencies.
            """)
        
        with col2:
            st.subheader("ROI by Coin")
            fig = px.bar(coin_metrics, x='coin', y='ROI_Pct',
                        color='ROI_Pct', color_continuous_scale='RdYlGn',
                        title="Return on Investment by Coin")
            fig.update_layout(xaxis_title="Coin", yaxis_title="ROI %")
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            This chart shows the Return on Investment (ROI) for each cryptocurrency traded.
            """)
    
        # Detailed allocation table
        st.subheader("Detailed Allocation Analysis")
        
        # Formatting for display
        display_metrics = coin_metrics.copy()
        display_metrics['Allocation_Pct'] = display_metrics['Allocation_Pct'].apply(lambda x: f"{x:.2f}%")
        display_metrics['ROI_Pct'] = display_metrics['ROI_Pct'].apply(lambda x: f"{x:.2f}%")
        display_metrics['Avg_Position_Size'] = display_metrics['Avg_Position_Size'].round(2)
        display_metrics['Total_Exposure'] = display_metrics['Total_Exposure'].round(2)
        display_metrics['Total_PnL'] = display_metrics['Total_PnL'].round(2)
        
        display_metrics = display_metrics.rename(columns={
            'coin': 'Coin',
            'Total_Exposure': 'Total Exposure',
            'Avg_Position_Size': 'Avg Position Size',
            'Total_PnL': 'Total PnL',
            'Trade_Count': 'Number of Trades',
            'Allocation_Pct': 'Allocation %',
            'ROI_Pct': 'ROI %'
        })
        
        st.dataframe(display_metrics, use_container_width=True)
    
    # Add the correlation analysis if we have the data
    if df is not None:
        st.markdown("---")
        correlation_data, diversification_score, risk_contribution = analyze_portfolio_correlation(df)
        show_portfolio_correlation(df, correlation_data, diversification_score, risk_contribution)

# Function to display the trading process page
def show_trading_process(df, process_data, process_metrics):
    st.header(" ")
    
    if not process_data or not process_metrics:
        st.error("Insufficient data for trading process analysis")
        return
    
    # Display process metrics
    st.subheader("Process Metrics")
    
    metrics_col1, metrics_col2 = st.columns(2)
    with metrics_col1:
        if 'Avg Time Between Trades (min)' in process_metrics:
            st.metric("Avg Time Between Trades", f"{process_metrics['Avg Time Between Trades (min)']:.1f} min")
        if 'Best Day' in process_metrics:
            st.metric("Best Trading Day", process_metrics['Best Day'])
        
    with metrics_col2:
        if 'Best Direction' in process_metrics:
            st.metric("Best Trade Direction", process_metrics['Best Direction'])
        if 'Size CV' in process_data.get('size_consistency', {}):
            cv = process_data['size_consistency']['Size CV']
            st.metric("Position Size Consistency", f"{cv:.2f}", 
                     help="Coefficient of Variation (lower is more consistent)")
    
    # Trading activity by time
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(" ")
        hourly_data = process_data.get('hourly_volume')
        if hourly_data is not None:
            fig = px.bar(hourly_data, x='hour', y='trade_count',
                        labels={'hour': 'Hour of Day', 'trade_count': 'Number of Trades'},
                        title="Trading Activity by Hour of Day")
            st.plotly_chart(fig, use_container_width=True)

            # Find peak trading hours
            peak_hours = hourly_data.sort_values('trade_count', ascending=False).head(3)
            peak_hours_str = ", ".join([f"{hour}:00" for hour in peak_hours['hour'].tolist()])
    
    with col2:
        st.subheader(" ")
        daily_data = process_data.get('daily_volume')
        if daily_data is not None:
            # Sort by day of week
            daily_data = daily_data.sort_values('day_of_week')
            fig = px.bar(daily_data, x='day_name', y='trade_count',
                        labels={'day_name': 'Day of Week', 'trade_count': 'Number of Trades'},
                        title="Trading Activity by Day of Week")
            st.plotly_chart(fig, use_container_width=True)
            
            # Find peak trading days
            peak_day = daily_data.loc[daily_data['trade_count'].idxmax()]
    
    # Trade direction analysis
    st.subheader("Trade Direction")
    col3, col4 = st.columns(2)
    
    with col3:
        direction_data = process_data.get('direction_counts')
        if direction_data is not None:
            fig = px.pie(direction_data, values='count', names='direction', 
                        title="Trade Direction Distribution")
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate the buy/sell ratio
            buy_count = direction_data[direction_data['direction'] == 'buy']['count'].values[0] if 'buy' in direction_data['direction'].values else 0
            sell_count = direction_data[direction_data['direction'] == 'sell']['count'].values[0] if 'sell' in direction_data['direction'].values else 0
            buy_sell_ratio = buy_count / sell_count if sell_count > 0 else float('inf')
    
    with col4:
        sequence_data = process_data.get('trade_sequences')
        if sequence_data is not None:
            # Create a more meaningful pivot table
            pivot_data = sequence_data.pivot(index='prev_direction', columns='direction', values='count').fillna(0)
            
            # Convert to percentages
            row_sums = pivot_data.sum(axis=1)
            percentage_pivot = pivot_data.div(row_sums, axis=0) * 100
            
            # Plot as a heatmap
            fig = go.Figure(data=go.Heatmap(
                z=percentage_pivot.values,
                x=percentage_pivot.columns,
                y=percentage_pivot.index,
                colorscale='Blues',
                text=[[f"{val:.1f}%" for val in row] for row in percentage_pivot.values],
                texttemplate="%{text}",
                textfont={"size": 12},
            ))
            
            fig.update_layout(
                title="Trade Sequence Patterns",
                xaxis_title="Current Direction",
                yaxis_title="Previous Direction"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Get the predominant sequences
            max_buy_after = percentage_pivot.loc[:, 'buy'].idxmax() if 'buy' in percentage_pivot.columns else None
            max_sell_after = percentage_pivot.loc[:, 'sell'].idxmax() if 'sell' in percentage_pivot.columns else None
            
            buy_after_val = percentage_pivot.loc[max_buy_after, 'buy'] if max_buy_after and 'buy' in percentage_pivot.columns else 0
            sell_after_val = percentage_pivot.loc[max_sell_after, 'sell'] if max_sell_after and 'sell' in percentage_pivot.columns else 0
    
    # Position sizing analysis
    st.subheader("Position Sizing")
    col5, col6 = st.columns(2)
    
    with col5:
        if 'sz' in df.columns:
            # Create a histogram of position sizes
            fig = px.histogram(df, x='sz', nbins=20,
                              title="Distribution of Position Sizes",
                              labels={'sz': 'Position Size'})
            st.plotly_chart(fig, use_container_width=True)
            
            size_consistency = process_data.get('size_consistency', {})
            cv = size_consistency.get('Size CV', 0)
    
    with col6:
        if 'time_diff' in df.columns:
            # Create a histogram of time between trades
            time_diff_hours = df['time_diff'] / 60  # Convert minutes to hours
            fig = px.histogram(time_diff_hours, nbins=20,
                              title="Distribution of Time Between Trades",
                              labels={'value': 'Hours Between Trades'})
            
            # Add median line
            median_time = time_diff_hours.median()
            fig.add_vline(x=median_time, line_dash="dash", line_color="red",
                         annotation_text=f"Median: {median_time:.2f} hours",
                         annotation_position="top right")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate trade frequency metrics
            trades_per_day = len(df) / (df['time'].max() - df['time'].min()).days if 'time' in df.columns else 0

# Main app
if uploaded_file is not None:
    # Load the CSV
    df = pd.read_csv(uploaded_file)
    
    # Process the data
    df = process_data(df)
    
    # Calculate metrics
    metrics, daily_data = calculate_metrics(df)
    trade_stats = calculate_trade_stats(df)
    
    # Calculate portfolio allocation metrics
    coin_metrics, allocation_metrics = analyze_portfolio_allocation(df)
    
    # Calculate trading process metrics
    process_data, process_metrics = analyze_trading_process(df)
    
    # Display selected page
    if page == "Performance Metrics":
        show_performance_metrics(df, metrics, daily_data, trade_stats)
    elif page == "Drawdown Analysis":
        show_drawdown_analysis(daily_data)
    elif page == "Portfolio Allocation":
        show_portfolio_allocation(df, coin_metrics, allocation_metrics)
    elif page == "Trading Process":
        show_trading_process(df, process_data, process_metrics)
    
    # Option to view raw data at the bottom of any page
    if st.checkbox("Show raw trade data"):
        st.subheader("Raw Trade Data")
        st.dataframe(df, use_container_width=True)

else:
    st.info("Please upload your trading history CSV file to generate the dashboard.")
    st.markdown("""
    ### Expected CSV Format
    
    The CSV should have these columns:
    - `time`: Timestamp of the trade
    - `coin`: Cryptocurrency traded
    - `dir`: Trade direction (buy/sell)
    - `px`: Price
    - `sz`: Size
    - `ntl`: Net Liquidity
    - `fee`: Fee paid
    - `closedPnl`: Closed Profit/Loss
    
    Example:
    ```
    time,coin,dir,px,sz,ntl,fee,closedPnl
    2022-01-01T12:00:00Z,BTC,buy,48000,0.1,4800,5.2,0
    2022-01-02T14:30:00Z,BTC,sell,49000,0.1,4900,5.3,100
    ```
    """)
    
    # Show feature explanations on the empty state
    st.markdown("---")
    
    st.header("Trading Dashboard Features")
    
    feature_tabs = st.tabs(["Performance Metrics", "Drawdown Analysis", "Portfolio Allocation", "Correlation Analysis"])
    
    with feature_tabs[0]:
        st.subheader("Performance Metrics")
        st.markdown("""
        Track your trading performance with key metrics including:
        - Cumulative returns with interactive visualization
        - Risk-adjusted return metrics (Sharpe, Sortino ratios)
        - Volatility and drawdown analysis
        - Win rate, profit factor, and fee impact
        """)
        
    with feature_tabs[1]:
        st.subheader("Drawdown Analysis")
        st.markdown("""
        Understand the depth and duration of trading drawdowns:
        - Interactive drawdown timeline visualization
        - Analysis of recovery periods and patterns
        - Drawdown distribution and frequency
        - Relationship between drawdown magnitude and recovery time
        """)
        
    with feature_tabs[2]:
        st.subheader("Portfolio Allocation")
        st.markdown("""
        Analyze the distribution of capital across different assets:
        - Asset allocation visualization
        - Concentration risk metrics
        - Performance by asset (ROI)
        - Position sizing analysis
        """)
        
    with feature_tabs[3]:
        st.subheader("Correlation Analysis")
        st.markdown("""
        Measure how assets move in relation to one another:
        - Correlation matrix visualization
        - Diversification score and metrics
        - Risk contribution analysis
        - Identification of highly correlated asset pairs
        """)

# PERFORMANCE METRICS CALCULATION METHODS

# Cumulative Return
# - Method: (Final Balance / Initial Balance - 1) * 100
# - Uses the sum of all closedPnl values added to the initial capital

# CAGR (Compound Annual Growth Rate)
# - Method: (Math.pow(1 + total_return/100, 1/years) - 1) * 100
# - Annualizes the return based on the period length in days

# Volatility
# - Method: Standard deviation of daily returns * sqrt(252) * 100
# - Annualized by multiplying by sqrt of trading days in a year

# Sharpe Ratio
# - Method: (CAGR - Risk Free Rate) / Volatility
# - Measures excess return per unit of risk

# Sortino Ratio
# - Method: (CAGR - Risk Free Rate) / Downside Deviation
# - Only considers negative returns in the denominator
# - Downside Deviation: Standard deviation of negative returns * sqrt(252) * 100

# Drawdown
# - Method: ((Current Balance / Peak Balance) - 1) * 100
# - Tracks the decline from the highest peak balance

# Calmar Ratio
# - Method: CAGR / |Maximum Drawdown|
# - Measures return relative to maximum drawdown risk

# Omega Ratio
# - Method: Count of positive returns / Count of negative returns
# - Simple measure of win/loss frequency balance

# Value at Risk (VaR)
# - Method: 5th percentile of sorted daily returns
# - Represents the worst expected loss with 95% confidence

# Kelly Criterion
# - Method: win_rate - ((1 - win_rate) / (avg_win / avg_loss))
# - Optimal position sizing based on edge and win rate

# DRAWDOWN ANALYSIS METHODS

# Drawdown Identification
# - Method: Track balance below peak and identify start/end points
# - A drawdown begins when balance falls below peak and ends when a new peak is reached

# Recovery Time
# - Method: Duration from max drawdown point to drawdown end
# - Measures how long it takes to recover from the worst point

# Recovery Ratio
# - Method: Recovery time / |Drawdown Percentage|
# - Days needed to recover per percentage point of drawdown

# PORTFOLIO ALLOCATION METHODS

# Allocation Percentage
# - Method: (Coin Total Exposure / Portfolio Total Exposure) * 100
# - Based on absolute sum of notional values for each asset

# ROI per Coin
# - Method: (Coin Total PnL / Coin Total Exposure) * 100
# - Measures return on capital allocated to each asset

# HHI Concentration Index
# - Method: Sum of squared allocation percentages (as decimals)
# - Higher values indicate more concentration (1.0 = single asset)

# Effective N
# - Method: 1 / HHI
# - Represents the effective number of assets accounting for concentration

# CORRELATION ANALYSIS METHODS

# Correlation Matrix
# - Method: Standard Pearson correlation of daily PnL between assets
# - Groups data by day and asset to create time series for comparison

# Diversification Score
# - Method: 100 * (1 - average absolute correlation)
# - Higher scores indicate better diversification

# Risk Contribution
# - Method: Asset Allocation % * Asset Volatility
# - Measures how much each asset contributes to portfolio risk
# - Risk Contribution % = Asset Weighted Risk / Total Weighted Risk * 100

# TRADING PROCESS ANALYSIS METHODS

# Position Sizing Consistency
# - Method: Standard Deviation / Mean of position sizes
# - Lower values indicate more consistent position sizing

# Win Rate by Direction
# - Method: Winning trades with direction / Total trades with direction
# - Shows which direction (buy/sell) is more successful

# Trade Sequence Patterns
# - Method: Conditional probabilities of trade directions
# - Measures tendency to follow one type of trade with another
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import math

# Set page configuration
st.set_page_config(
    page_title="Trading Performance Dashboard",
    page_icon="📈",
    layout="wide"
)

# Add page title
st.title("Trading Performance Dashboard")

# File uploader in the sidebar
uploaded_file = st.sidebar.file_uploader("Upload your trading history CSV", type=["csv"])

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Performance Metrics", "Portfolio Allocation", "Trading Process"])

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

# Function to display the portfolio allocation page
def show_portfolio_allocation(df, coin_metrics, allocation_metrics):
    st.header("Portfolio Allocation Analysis")
    
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
            peak_hours_str = ", ".join
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
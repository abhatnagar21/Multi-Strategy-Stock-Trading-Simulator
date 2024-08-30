import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
import numpy as np

# Set the default theme to dark
pio.templates.default = "plotly_dark"

# Fetching TCS stock data for a specific period
ticker_symbol = "TCS.NS"
data = yf.download(ticker_symbol, start="2022-01-01", end="2024-01-02")

# Parameters for the strategy
rolling_window = 14
stop_loss_pct = 0.05  # 5% stop loss
take_profit_pct = 0.10  # 10% take profit
transaction_fee_pct = 0.001  # 0.1% per transaction

# Calculate Support and Resistance using rolling windows
data['Support'] = data['Low'].rolling(window=rolling_window).min()
data['Resistance'] = data['High'].rolling(window=rolling_window).max()

# Calculate RSI (Relative Strength Index)
def calculate_rsi(series, window=14):
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

data['RSI'] = calculate_rsi(data['Close'], window=rolling_window)

# Calculate Bollinger Bands
data['Middle Band'] = data['Close'].rolling(window=rolling_window).mean()
data['Upper Band'] = data['Middle Band'] + 2 * data['Close'].rolling(window=rolling_window).std()
data['Lower Band'] = data['Middle Band'] - 2 * data['Close'].rolling(window=rolling_window).std()

# Initialize variables for the trading strategy
def initialize_strategy_vars():
    return {
        'cash': initial_cash,
        'stock_held': 0,
        'portfolio_value': initial_cash,
        'trade_log': [],
        'buy_signals': [],
        'sell_signals': [],
        'entry_price': 0,
        'win_trades': 0,
        'loss_trades': 0,
        'max_drawdown': 0,
        'peak_value': initial_cash
    }

initial_cash = 100000  # Initial cash in the portfolio
strategies = {
    'Support/Resistance': initialize_strategy_vars(),
    'RSI': initialize_strategy_vars(),
    'Bollinger Bands': initialize_strategy_vars(),
}

# Define buy and sell logic for each strategy
def buy_logic(strategy_vars, current_price, signal):
    if strategy_vars['cash'] > 0:
        strategy_vars['stock_held'] = strategy_vars['cash'] / current_price * (1 - transaction_fee_pct)
        strategy_vars['cash'] = 0
        strategy_vars['portfolio_value'] = strategy_vars['stock_held'] * current_price
        strategy_vars['entry_price'] = current_price
        strategy_vars['buy_signals'].append((data.index[i], current_price))
        strategy_vars['trade_log'].append(f"Bought at {current_price:.2f} on {data.index[i]} using {signal}")

def sell_logic(strategy_vars, current_price, signal):
    if strategy_vars['stock_held'] > 0:
        strategy_vars['cash'] = strategy_vars['stock_held'] * current_price * (1 - transaction_fee_pct)
        strategy_vars['stock_held'] = 0
        strategy_vars['portfolio_value'] = strategy_vars['cash']
        strategy_vars['sell_signals'].append((data.index[i], current_price))
        strategy_vars['trade_log'].append(f"Sold at {current_price:.2f} on {data.index[i]} using {signal}")
        if current_price >= strategy_vars['entry_price']:
            strategy_vars['win_trades'] += 1
        else:
            strategy_vars['loss_trades'] += 1

# Execute the trading strategies
for i in range(1, len(data)):
    current_price = data['Close'][i]
    
    for strategy_name, strategy_vars in strategies.items():
        # Update peak value and drawdown
        strategy_vars['portfolio_value'] = strategy_vars['cash'] + strategy_vars['stock_held'] * current_price
        strategy_vars['peak_value'] = max(strategy_vars['peak_value'], strategy_vars['portfolio_value'])
        drawdown = (strategy_vars['peak_value'] - strategy_vars['portfolio_value']) / strategy_vars['peak_value']
        strategy_vars['max_drawdown'] = max(strategy_vars['max_drawdown'], drawdown)

        # Strategy specific logic
        if strategy_name == 'Support/Resistance':
            if current_price > data['Resistance'][i-1] and strategy_vars['cash'] > 0:
                buy_logic(strategy_vars, current_price, strategy_name)
            elif current_price < data['Support'][i-1] and strategy_vars['stock_held'] > 0:
                sell_logic(strategy_vars, current_price, strategy_name)

        elif strategy_name == 'RSI':
            if data['RSI'][i] < 30 and strategy_vars['cash'] > 0:
                buy_logic(strategy_vars, current_price, strategy_name)
            elif data['RSI'][i] > 70 and strategy_vars['stock_held'] > 0:
                sell_logic(strategy_vars, current_price, strategy_name)

        elif strategy_name == 'Bollinger Bands':
            if current_price < data['Lower Band'][i-1] and strategy_vars['cash'] > 0:
                buy_logic(strategy_vars, current_price, strategy_name)
            elif current_price > data['Upper Band'][i-1] and strategy_vars['stock_held'] > 0:
                sell_logic(strategy_vars, current_price, strategy_name)

# Final portfolio values
final_values = {strategy_name: strategy_vars['cash'] + strategy_vars['stock_held'] * data['Close'].iloc[-1] for strategy_name, strategy_vars in strategies.items()}

# Performance metrics
performance_metrics = {
    strategy_name: {
        'Final Portfolio Value': final_values[strategy_name],
        'Profit/Loss': final_values[strategy_name] - initial_cash,
        'Total Trades': strategy_vars['win_trades'] + strategy_vars['loss_trades'],
        'Win Ratio': strategy_vars['win_trades'] / (strategy_vars['win_trades'] + strategy_vars['loss_trades']) if strategy_vars['win_trades'] + strategy_vars['loss_trades'] > 0 else 0,
        'Max Drawdown': strategy_vars['max_drawdown']
    } for strategy_name, strategy_vars in strategies.items()
}

# Print performance metrics for each strategy
for strategy_name, metrics in performance_metrics.items():
    print(f"\n{strategy_name} Strategy:")
    for key, value in metrics.items():
        print(f"{key}: {value:.2f}" if isinstance(value, (float, int)) else f"{key}: {value}")

# Function to create a graph for a strategy
def create_strategy_graph(strategy_name, strategy_vars):
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Candlesticks'
    ))
    
    # Add strategy specific indicators and signals
    if strategy_name == 'Support/Resistance':
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Support'],
            mode='lines',
            line=dict(color='green', width=1),
            name='Support'
        ))
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Resistance'],
            mode='lines',
            line=dict(color='red', width=1),
            name='Resistance'
        ))

    elif strategy_name == 'RSI':
        fig.add_trace(go.Scatter(
            x=data.index,
            y=[30] * len(data.index),
            mode='lines',
            line=dict(color='orange', width=1, dash='dash'),
            name='RSI 30'
        ))
        fig.add_trace(go.Scatter(
            x=data.index,
            y=[70] * len(data.index),
            mode='lines',
            line=dict(color='orange', width=1, dash='dash'),
            name='RSI 70'
        ))
        
    elif strategy_name == 'Bollinger Bands':
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Middle Band'],
            mode='lines',
            line=dict(color='blue', width=1),
            name='Middle Band'
        ))
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Upper Band'],
            mode='lines',
            line=dict(color='yellow', width=1, dash='dash'),
            name='Upper Band'
        ))
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Lower Band'],
            mode='lines',
            line=dict(color='white', width=1, dash='dash'),
            name='Lower Band'
        ))
    
    # Add Buy/Sell Signals
    if strategy_vars['buy_signals']:
        fig.add_trace(go.Scatter(
            x=[signal[0] for signal in strategy_vars['buy_signals']],
            y=[signal[1] for signal in strategy_vars['buy_signals']],
            mode='markers',
            marker=dict(color='green', size=10, symbol='triangle-up'),
            name='Buy Signals'
        ))
        
    if strategy_vars['sell_signals']:
        fig.add_trace(go.Scatter(
            x=[signal[0] for signal in strategy_vars['sell_signals']],
            y=[signal[1] for signal in strategy_vars['sell_signals']],
            mode='markers',
            marker=dict(color='red', size=10, symbol='triangle-down'),
            name='Sell Signals'
        ))

    # Update layout and show plot
    fig.update_layout(
        title=f"{strategy_name} Strategy for {ticker_symbol}",
        yaxis_title="Price",
        xaxis_title="Date",
        xaxis_rangeslider_visible=False
    )
    fig.show()

# Create and display the strategy graphs
for strategy_name, strategy_vars in strategies.items():
    create_strategy_graph(strategy_name, strategy_vars)
# Print performance metrics for each strategy
for strategy_name, metrics in performance_metrics.items():
    print(f"\n{strategy_name} Strategy:")
    for key, value in metrics.items():
        print(f"{key}: {value:.2f}" if isinstance(value, (float, int)) else f"{key}: {value}")

    # Print the trade log (buying and selling prices)
    print("\nTrade Log:")
    for log in strategies[strategy_name]['trade_log']:
        print(log)

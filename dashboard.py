import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import psutil
import matplotlib.pyplot as plt
import seaborn as sns


def generate_mock_data():
    # 生成日期范围
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*3)
    dates = pd.date_range(start_date, end_date)

    # 生成随机收益率序列
    np.random.seed(42)
    daily_returns = np.random.normal(0.001, 0.02, len(dates))
    cumulative_returns = (1 + daily_returns).cumprod() - 1

    # 生成持仓数据
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'JPM', 'GS', 'BTC-USD', 'ETH-USD']
    positions = {sym: np.random.randint(1, 100) for sym in symbols}
    prices = {sym: np.random.uniform(10, 1000) for sym in symbols}
    sectors = {
        'AAPL': '科技', 'MSFT': '科技', 'GOOGL': '科技',
        'AMZN': '消费', 'TSLA': '汽车', 'NVDA': '半导体',
        'JPM': '金融', 'GS': '金融', 'BTC-USD': '加密货币', 'ETH-USD': '加密货币'
    }

    # 生成交易历史
    trades = []
    for _ in range(100):
        sym = np.random.choice(symbols)
        direction = np.random.choice(['Buy', 'Sell'])
        qty = np.random.randint(1, 50)
        price = prices[sym] * np.random.uniform(0.95, 1.05)
        cost = price * qty * 0.001  # 0.1%交易成本
        slippage = price * 0.0005 * np.random.randn()
        trades.append({
            'Date': np.random.choice(dates),
            'Symbol': sym,
            'Direction': direction,
            'Quantity': qty,
            'Price': price,
            'Cost': cost,
            'Slippage': slippage
        })
    trades_df = pd.DataFrame(trades)

    return {
        'dates': dates,
        'daily_returns': daily_returns,
        'cumulative_returns': cumulative_returns,
        'positions': positions,
        'prices': prices,
        'sectors': sectors,
        'trades': trades_df,
        'symbols': symbols
    }


def calculate_metrics(data, daily_returns, cumulative_returns, trades_df, risk_free_rate=0.02):
    # 基本指标
    total_return = cumulative_returns[-1]
    annualized_return = (1 + total_return) ** (365/len(daily_returns)) - 1
    volatility = np.std(daily_returns) * np.sqrt(252)
    sharpe_ratio = (np.mean(daily_returns) * 252 - risk_free_rate) / volatility

    # 最大回撤
    peak = cumulative_returns[0]
    max_drawdown = 0
    for ret in cumulative_returns:
        if ret > peak:
            peak = ret
        dd = (peak - ret) / (1 + peak)
        if dd > max_drawdown:
            max_drawdown = dd

    # 交易相关指标
    win_trades = trades_df[trades_df['Direction'] == 'Sell']
    win_rate = len(win_trades) / len(trades_df) if len(trades_df) > 0 else 0
    avg_profit = win_trades['Price'].mean() if len(win_trades) > 0 else 0
    loss_trades = trades_df[trades_df['Direction'] == 'Buy']
    avg_loss = loss_trades['Price'].mean() if len(loss_trades) > 0 else 0
    profit_loss_ratio = avg_profit / avg_loss if avg_loss != 0 else 0

    # VaR计算 (95%置信水平)
    var_95 = np.percentile(daily_returns, 5)

    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_loss_ratio': profit_loss_ratio,
        'volatility': volatility,
        'var_95': var_95,
        'trade_frequency': len(trades_df) / (len(data['dates']) / 252),  # 年化交易频率
        'avg_daily_trades': len(trades_df) / len(data['dates'])
    }


def calculate_position_metrics(positions, prices, sectors):
    total_value = sum(positions[sym] * prices[sym] for sym in positions)
    sector_exposure = {}
    for sym in positions:
        sector = sectors[sym]
        value = positions[sym] * prices[sym]
        sector_exposure[sector] = sector_exposure.get(sector, 0) + value

    # 转换为百分比
    for sector in sector_exposure:
        sector_exposure[sector] = sector_exposure[sector] / total_value * 100

    # 集中度风险 (最大持仓占比)
    position_values = [positions[sym] * prices[sym] for sym in positions]
    concentration_risk = max(position_values) / total_value if total_value > 0 else 0

    return {
        'total_value': total_value,
        'sector_exposure': sector_exposure,
        'concentration_risk': concentration_risk,
        'leverage': 1.0,  # 简化假设
        'long_exposure': 0.6,  # 简化假设
        'short_exposure': 0.4   # 简化假设
    }


def calculate_execution_metrics(trades_df):
    total_cost = trades_df['Cost'].sum()
    total_slippage = trades_df['Slippage'].sum()
    total_turnover = (trades_df['Quantity'] * trades_df['Price']).sum()
    cost_ratio = total_cost / total_turnover if total_turnover > 0 else 0
    slippage_ratio = total_slippage / total_turnover if total_turnover > 0 else 0

    # 假设有100个订单，90个成交
    fill_rate = 0.9

    return {
        'total_cost': total_cost,
        'total_slippage': total_slippage,
        'cost_ratio': cost_ratio,
        'slippage_ratio': slippage_ratio,
        'fill_rate': fill_rate
    }


def get_system_metrics():
    return {
        'cpu_usage': psutil.cpu_percent(),
        'memory_usage': psutil.virtual_memory().percent,
        'latency': np.random.uniform(5, 20),  # 毫秒
        'alerts': []  # 可以添加警报逻辑
    }


def setup_page_layout():
    """设置页面布局"""
    st.set_page_config(layout="wide", page_title="交易绩效仪表盘")
    st.title("交易绩效仪表盘")


def display_sidebar(data, metrics, position_metrics, system_metrics):
    """显示侧边栏内容"""
    with st.sidebar:
        st.header("控制面板")
        auto_refresh = st.checkbox('自动刷新 (5秒)', value=True)

        st.header("警报系统")
        if metrics['max_drawdown'] > 0.1:
            st.error(f"⚠️ 最大回撤超过阈值: {metrics['max_drawdown'] * 100:.2f}%")
        if position_metrics['concentration_risk'] > 0.3:
            st.error(f"⚠️ 集中度风险过高: {position_metrics['concentration_risk'] * 100:.2f}%")
        if system_metrics['cpu_usage'] > 80:
            st.error(f"⚠️ CPU使用率过高: {system_metrics['cpu_usage']}%")
        if system_metrics['latency'] > 15:
            st.warning(f"⚠️ 系统延迟较高: {system_metrics['latency']:.2f} ms")

        st.header("报告导出")
        if st.button("生成PDF报告"):
            st.success("报告生成中... (模拟功能)")

    return auto_refresh


def display_main_content(data, metrics, position_metrics, execution_metrics, system_metrics):
    """显示主内容区域"""
    # 1. 关键指标概览
    st.header("📊 关键绩效指标")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("累计收益率", f"{metrics['total_return'] * 100:.2f}%")
    col2.metric("年化收益率", f"{metrics['annualized_return'] * 100:.2f}%")
    col3.metric("夏普比率", f"{metrics['sharpe_ratio']:.2f}")
    col4.metric("最大回撤", f"{metrics['max_drawdown'] * 100:.2f}%")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("胜率", f"{metrics['win_rate'] * 100:.2f}%")
    col6.metric("盈亏比", f"{metrics['profit_loss_ratio']:.2f}")
    col7.metric("年化波动率", f"{metrics['volatility'] * 100:.2f}%")
    col8.metric("VaR (95%)", f"{metrics['var_95'] * 100:.2f}%")

    # 2. 收益与风险图表
    st.header("📈 收益与风险分析")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=data['dates'], y=data['cumulative_returns'], name="累计收益"))
    fig1.update_layout(title="累计收益率曲线", xaxis_title="日期", yaxis_title="收益率")
    st.plotly_chart(fig1, use_container_width=True)

    # 回撤曲线
    fig2 = go.Figure()
    peak = data['cumulative_returns'][0]
    drawdowns = []
    for ret in data['cumulative_returns']:
        if ret > peak:
            peak = ret
        drawdown = (peak - ret) / (1 + peak)
        drawdowns.append(drawdown)
    fig2.add_trace(go.Scatter(x=data['dates'], y=drawdowns, name="回撤", line=dict(color='red')))
    fig2.update_layout(title="回撤曲线", xaxis_title="日期", yaxis_title="回撤幅度")
    st.plotly_chart(fig2, use_container_width=True)

    # 3. 持仓分析
    st.header("🏦 持仓分析")
    col9, col10, col11 = st.columns(3)
    col9.metric("总持仓价值", f"${position_metrics['total_value']:,.2f}")
    col10.metric("集中度风险", f"{position_metrics['concentration_risk'] * 100:.2f}%")
    col11.metric("杠杆率", f"{position_metrics['leverage']:.2f}x")

    # 行业分布
    st.subheader("行业分布")
    sector_df = pd.DataFrame.from_dict(position_metrics['sector_exposure'], orient='index', columns=['Exposure'])
    fig3 = px.pie(sector_df, values='Exposure', names=sector_df.index, title="行业分布")
    st.plotly_chart(fig3, use_container_width=True)

    # 持仓明细
    st.subheader("持仓明细")
    positions_df = pd.DataFrame({
        'Symbol': list(data['positions'].keys()),
        'Quantity': list(data['positions'].values()),
        'Price': [data['prices'][sym] for sym in data['positions']],
        'Sector': [data['sectors'][sym] for sym in data['positions']]
    })
    positions_df['Value'] = positions_df['Quantity'] * positions_df['Price']
    positions_df['Weight'] = positions_df['Value'] / position_metrics['total_value'] * 100
    st.dataframe(positions_df.sort_values('Value', ascending=False))

    # 4. 交易执行分析
    st.header("⚙️ 交易执行分析")
    col12, col13, col14 = st.columns(3)
    col12.metric("总交易成本", f"${execution_metrics['total_cost']:,.2f}")
    col13.metric("成本占比", f"{execution_metrics['cost_ratio'] * 100:.4f}%")
    col14.metric("滑点占比", f"{execution_metrics['slippage_ratio'] * 100:.4f}%")

    # 交易频率
    st.subheader("交易频率")
    fig4 = px.histogram(data['trades'], x='Date', title="交易频率分布")
    st.plotly_chart(fig4, use_container_width=True)

    # 5. 系统监控
    st.header("🖥️ 系统监控")
    col15, col16, col17 = st.columns(3)
    col15.metric("CPU使用率", f"{system_metrics['cpu_usage']}%")
    col16.metric("内存使用率", f"{system_metrics['memory_usage']}%")
    col17.metric("平均延迟", f"{system_metrics['latency']:.2f} ms")

    # 6. 相关性分析
    st.header("🔗 相关性分析")
    # 生成模拟相关性矩阵
    corr_matrix = pd.DataFrame(np.random.uniform(-1, 1, size=(len(data['symbols']), len(data['symbols']))),
                               index=data['symbols'], columns=data['symbols'])
    np.fill_diagonal(corr_matrix.values, 1)

    fig5 = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu',
        zmin=-1,
        zmax=1))
    fig5.update_layout(title="持仓标的相关性矩阵")
    st.plotly_chart(fig5, use_container_width=True)

    # 7. 市场状态指标
    st.header("🌍 市场状态")
    col18, col19, col20 = st.columns(3)
    col18.metric("市场波动率 (模拟VIX)", "22.5")
    col19.metric("平均买卖价差", "0.05%")
    col20.metric("市场流动性指数", "85.2")

    # 8. 资金管理
    st.header("💰 资金管理")
    col21, col22, col23 = st.columns(3)
    col21.metric("可用资金", "$1,250,000")
    col22.metric("保证金使用率", "35.2%")
    col23.metric("当日盈亏", "$12,450")


def main():
    """主函数"""
    # 初始化数据
    if 'data' not in st.session_state:
        st.session_state.data = generate_mock_data()
    data = st.session_state.data

    # 计算各项指标
    metrics = calculate_metrics(data, data['daily_returns'], data['cumulative_returns'], data['trades'])
    position_metrics = calculate_position_metrics(data['positions'], data['prices'], data['sectors'])
    execution_metrics = calculate_execution_metrics(data['trades'])
    system_metrics = get_system_metrics()

    # 设置页面布局
    setup_page_layout()

    # 显示侧边栏并获取控制参数
    auto_refresh = display_sidebar(data, metrics, position_metrics, system_metrics)

    # 显示主内容
    display_main_content(data, metrics, position_metrics, execution_metrics, system_metrics)

    # 自动刷新逻辑
    if auto_refresh:
        time.sleep(5)
        st.rerun()


if __name__ == "__main__":
    main()
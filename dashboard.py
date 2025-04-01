import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests



# 生成模拟交易数据
def generate_fake_trades():
    dates = pd.date_range(start="2024-01-01", periods=50, freq='D')
    returns = np.random.randn(50).cumsum()
    df = pd.DataFrame({'Date': dates, 'Cumulative Returns': returns})
    return df


# 交易绩效指标计算
def calculate_performance():
    win_rate = np.random.uniform(40, 70)
    sharpe_ratio = np.random.uniform(1, 3)
    max_drawdown = np.random.uniform(-15, -5)
    return win_rate, sharpe_ratio, max_drawdown


# 模拟订单列表
def generate_order_list():
    orders = pd.DataFrame({
        '时间': pd.date_range(start='2024-03-01', periods=10, freq='H'),
        '交易对': np.random.choice(['BTC/USDT', 'ETH/USDT', 'BNB/USDT'], 10),
        '方向': np.random.choice(['买入', '卖出'], 10),
        '价格': np.round(np.random.uniform(30000, 60000, 10), 2),
        '数量': np.round(np.random.uniform(0.1, 2, 10), 4)
    })
    return orders


# 获取市场情绪数据（示例：恐慌贪婪指数）
def get_market_sentiment():
    return np.random.uniform(0, 100)


# Streamlit 界面
def main():
    st.set_page_config(layout="wide", page_title="量化交易仪表盘")
    st.title("📈 量化交易仪表盘")

    col1, col2 = st.columns([2, 1])

    with col1:

        st.subheader("近期收益曲线")
        trades_data = generate_fake_trades()
        fig = px.line(trades_data, x='Date', y='Cumulative Returns', title='累计收益')
        st.plotly_chart(fig)

        st.subheader("订单列表")
        orders = generate_order_list()
        st.dataframe(orders)

    with col2:
        st.subheader("最新消息采集")
        st.write("🚀 这里可以集成新闻 API 来显示市场最新动态。")

        st.subheader("市场情绪指标")
        sentiment = get_market_sentiment()
        st.metric("恐慌贪婪指数", f"{sentiment:.2f}")

        st.subheader("交易绩效")
        win_rate, sharpe, drawdown = calculate_performance()
        st.metric("胜率", f"{win_rate:.2f}%")
        st.metric("夏普比率", f"{sharpe:.2f}")
        st.metric("最大回撤", f"{drawdown:.2f}%")


if __name__ == "__main__":
    main()

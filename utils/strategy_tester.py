"""
时序测试综合策略，给出相关数据。以下是衡量策略绩效的一些重要指标：

1. 累计回报率（Total Return）：策略在一段时间内的总回报率，包括资本增值和收益再投资。

2. 年化回报率（Annualized Return）：策略在一年内的平均回报率，用于比较不同策略或投资产品之间的绩效。

3. 夏普比率（Sharpe Ratio）：衡量单位风险所获得的超额回报。夏普比率越高，表明单位风险下获得的回报越高。

4. 最大回撤（Maximum Drawdown）：策略在历史最高点到最低点期间的最大损失幅度。衡量策略在市场下跌时的风险承受能力。

5. 胜率（Win Rate）：策略获得正回报的交易次数占总交易次数的比例。胜率越高，策略的交易方向越准确。

6. 平均获利/亏损比率（Average Profit/Loss Ratio）：获利交易的平均盈利与亏损交易的平均亏损之比。衡量获利能力与亏损能力的平衡性。

7. 平均持仓时间（Average Holding Period）：策略平均持有头寸的时间长度。衡量策略的交易频率和持仓期限。

8. Beta系数（Beta）：衡量策略相对于市场的敏感性。Beta系数大于1表示策略相对于市场具有更高的波动性，小于1表示具有较低的波动性。

9. Alpha系数（Alpha）：策略相对于市场表现的超额回报。Alpha为正值表示策略表现优于市场，为负值表示表现不如市场。

10. 调整后夏普比率（Adjusted Sharpe Ratio）：考虑到风险无风险利率的夏普比率。用于评估风险调整后的回报。

11. Sortino比率（Sortino Ratio）：衡量单位下行风险所获得的超额回报。Sortino比率越高，表明单位下行风险下获得的回报越高。

12. Calmar比率（Calmar Ratio）：策略年化回报率与最大回撤的比率。Calmar比率越高，表明策略的风险调整回报越好。

这些指标可以根据策略的特点和需求进行选择和组合使用，以全面评估策略的绩效表现。值得注意的是，这些指标并非独立评估策略的唯一标准，还需要考虑其他因素，如投资目标、风险承受能力和市场环境等。
"""


class StrategyTester:
    """
    测试一段时间内，策略的各项绩效指标
    """

    def __init__(self) -> None:
        pass

    def _total_return(self):
        pass

    # def _annualized_return(self):
    #     pass

    # def _sharpe_ratio(self):
    #     pass

    # def _max_drawdown(self):
    #     pass

    # def _win_rate(self):
    #     pass

    # def _profit_loss_ratio(self):
    #     pass

    # def _average_holding_period(self):
    #     pass

    # def _beta(self):
    #     pass

    # def _alpha(self):
    #     pass

    # def _adjusted_sharpe_ratio(self):
    #     pass

    # def _sortino_ratio(self):
    #     pass

    # def _calmar_ratio(self):
    #     pass

    def test(self):
        pass

"""
Visualization Dashboard

Plotly-based charts for strategy comparison results.
Can be rendered standalone or integrated into Trading_Simulator.
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from loguru import logger

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    go = None
    make_subplots = None

from ..comparison.strategy_comparator import ComparisonResult, StrategyMetrics
from ..risk_evaluation.risk_profile import RiskProfile


class Dashboard:
    """Generates Plotly charts from comparison results."""

    def __init__(self, output_dir: str = "results/charts"):
        if go is None:
            raise ImportError("plotly is required: pip install plotly")
        self.output_dir = output_dir
        self.logger = logger.bind(module="dashboard")

    def plot_equity_curves(
        self,
        result: ComparisonResult,
        title: str = "Strategy Comparison — Equity Curves",
        save: bool = True,
    ) -> go.Figure:
        """Plot NAV time series for all strategies."""
        fig = go.Figure()

        for name, nav in result.equity_curves.items():
            metrics = result.strategies.get(name)
            label = f"{name} (Sharpe: {metrics.sharpe_ratio:.2f})" if metrics else name

            fig.add_trace(go.Scatter(
                x=nav.index,
                y=nav.values,
                mode="lines",
                name=label,
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            hovermode="x unified",
            template="plotly_white",
        )

        if save:
            fig.write_html(f"{self.output_dir}/equity_curves.html")

        return fig

    def plot_drawdowns(
        self,
        result: ComparisonResult,
        title: str = "Strategy Drawdowns",
        save: bool = True,
    ) -> go.Figure:
        """Plot drawdown curves for all strategies."""
        fig = go.Figure()

        for name, nav in result.equity_curves.items():
            cummax = nav.cummax()
            drawdown = (nav - cummax) / cummax * 100

            fig.add_trace(go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                mode="lines",
                name=name,
                fill="tozeroy",
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            hovermode="x unified",
            template="plotly_white",
        )

        if save:
            fig.write_html(f"{self.output_dir}/drawdowns.html")

        return fig

    def plot_metrics_comparison(
        self,
        result: ComparisonResult,
        title: str = "Strategy Metrics Comparison",
        save: bool = True,
    ) -> go.Figure:
        """Bar chart comparing key metrics across strategies."""
        names = list(result.strategies.keys())
        metrics_list = list(result.strategies.values())

        metric_names = [
            "Total Return (%)", "Sharpe Ratio", "Max Drawdown (%)",
            "Volatility (%)", "Win Rate (%)", "Avg Positions",
        ]

        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=metric_names,
        )

        values_map = [
            [m.total_return * 100 for m in metrics_list],
            [m.sharpe_ratio for m in metrics_list],
            [m.max_drawdown * 100 for m in metrics_list],
            [m.volatility * 100 for m in metrics_list],
            [m.win_rate * 100 for m in metrics_list],
            [m.avg_positions for m in metrics_list],
        ]

        for i, (metric_vals, metric_name) in enumerate(zip(values_map, metric_names)):
            row = i // 3 + 1
            col = i % 3 + 1
            fig.add_trace(
                go.Bar(x=names, y=metric_vals, name=metric_name, showlegend=False),
                row=row, col=col,
            )

        fig.update_layout(
            title=title,
            template="plotly_white",
            height=600,
        )

        if save:
            fig.write_html(f"{self.output_dir}/metrics_comparison.html")

        return fig

    def plot_risk_heatmap(
        self,
        profiles: Dict[str, RiskProfile],
        title: str = "Stock Risk Heatmap",
        save: bool = True,
    ) -> go.Figure:
        """Heatmap of risk metrics across stocks."""
        tickers = list(profiles.keys())
        metrics = ["annualized_vol", "beta", "max_drawdown", "sharpe", "win_rate", "var_95"]
        labels = ["Vol (%)", "Beta", "Max DD (%)", "Sharpe", "Win Rate", "VaR 95"]

        data = []
        for metric in metrics:
            row = []
            for t in tickers:
                val = getattr(profiles[t], metric, 0)
                if metric in ("annualized_vol", "max_drawdown", "var_95"):
                    val *= 100
                row.append(val)
            data.append(row)

        fig = go.Figure(data=go.Heatmap(
            z=data,
            x=tickers,
            y=labels,
            colorscale="RdYlGn_r",
            text=[[f"{v:.1f}" for v in row] for row in data],
            texttemplate="%{text}",
        ))

        fig.update_layout(
            title=title,
            template="plotly_white",
            height=400,
        )

        if save:
            fig.write_html(f"{self.output_dir}/risk_heatmap.html")

        return fig

    def plot_allocation_pie(
        self,
        weights: Dict[str, float],
        strategy_name: str,
        save: bool = True,
    ) -> go.Figure:
        """Pie chart of portfolio allocation."""
        # Group small positions as "Other"
        threshold = 0.03
        main = {t: w for t, w in weights.items() if w >= threshold}
        other = sum(w for w in weights.values() if w < threshold)
        if other > 0:
            main["Other"] = other

        fig = go.Figure(data=[go.Pie(
            labels=list(main.keys()),
            values=list(main.values()),
            textinfo="label+percent",
        )])

        fig.update_layout(
            title=f"{strategy_name} — Portfolio Allocation",
            template="plotly_white",
        )

        if save:
            fig.write_html(f"{self.output_dir}/allocation_{strategy_name}.html")

        return fig

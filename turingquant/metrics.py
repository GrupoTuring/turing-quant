"""Módulo para metrificação de ativos e retornos."""

import numpy as np
import pandas as pd
import plotly.express as px


def sharpe_ratio(returns, risk_free=0):
    """
    Essa função, a partir da definição do parâmetro de retorno, fornece o sharpe ratio do ativo, com base na média histórica e desvio padrão dos retornos.
    O risk free considerado é nulo.

    Args:
        returns (pd.series): série com o retorno do ativo.
        risk_free (float): risk free utilizado para cálculo do sharpe ratio.

    Returns:
        float: índice de sharpe do ativo.
    """

    expected_returns = returns.mean()
    risk = returns.std()

    return(expected_returns - risk_free) / risk


def beta(returns, benchmark):
    """
    Essa função, a partir do fornecimento dos retornos do ativo e do benchmark, calcula o beta do ativo.

    Args:
        returns (pd.series): série com o retorno do ativo.
        benchmark (pd.series): série com o retorno do benchmark.

    Returns:
        float: Beta do ativo
    """
    concat = np.matrix([returns, benchmark])
    cov = np.cov(concat)[0][1]
    benchmark_vol = np.var(benchmark)

    return cov / benchmark_vol


def alpha(start_price, end_price, dps):
    """
    Essa função, com o fornecimento do preço final, dos dividendos por ação e do preço inicial, a calcula o alfa de um ativo.

    Args:
        start_price (float): preço inicial.
        end_price (float): preço final.
        dps(float): dividendos por ação.

    Returns:
        float: alpha do ativo
    """

    return(end_price + dps - start_price) / start_price


def drawdown(return_series, plot=True):
    """
    Calcula e plota o drawdown percentual para uma série de retornos.

    Args:
        return_series (pd.Series): série de retornos para o qual será calculado o Drawdown.
        plot (bool): se `True`, plota um gráfico de linha com o Drawdown ao longo do tempo.

    Returns:
        pd.Series: uma série com os valores percentuais do Drawdown.
    """

    wealth_index = 1000 * (1 + return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = pd.Series((wealth_index - previous_peaks)/previous_peaks, name='Drawdown')

    if plot:
        fig = px.line(drawdowns, x=drawdowns.index, y='Drawdown', title='Drawdown')
        fig.update_xaxes(title_text='Tempo')
        fig.update_yaxes(title_text='Drawdown (%)')
        fig.show()

    return drawdowns


def rolling_beta(returns, benchmark, window, plot=True):
    """
    Plota o beta móvel para um ativo e um benchmark de referência, na forma de séries de retornos.

    Args:
        returns (array): série de retornos para o qual o beta será calculado.
        benchmark (array): série de retornos para usar de referência no cálculo do beta.
        window (int): janela móvel para calcular o beta ao longo do tempo.
        plot (bool): se `True`, plota um gráfico de linha com o beta ao longo do tempo.

    Returns:
        pd.Series: uma série com os valores do Beta para os últimos `window` dias.
        A série não possui os `window` primeiros dias.

    """
    rolling_beta = pd.Series([beta(returns[i-window:i], benchmark[i-window:i])
                             for i in range(window, len(returns))], index=returns[window:].index)
    if plot:
        fig = px.line(rolling_beta, title="Beta móvel")
        overall_beta = beta(returns, benchmark)
        fig.update_layout(shapes=[
            dict(
                type='line',
                xref='paper', x0=0, x1=1,
                yref='y', y0=overall_beta, y1=overall_beta,
                line=dict(
                    color='grey',
                    width=2,
                    dash='dash'
                )
            )
        ], annotations=[
            dict(
                text='beta total: %.3f' % overall_beta,
                xref='paper', x=0.05,
                yref='y', y=overall_beta,
                xanchor='left'
            )
        ])
        fig.update_layout(showlegend=False)
        fig.update_xaxes(title_text='Tempo')
        fig.update_yaxes(title_text='Beta móvel: ' + str(window) + ' períodos')
        fig.show()
    return rolling_beta


def rolling_sharpe(returns, window, risk_free=0, plot=True):
    """
    Plota o sharpe móvel para um ativo e um benchmark de referência, na forma de séries de retornos.

    Args:
        returns (array): série de retornos para o qual o Sharpe Ratio será calculado.
        window (int): janela móvel para calcular o Sharpe ao longo do tempo.
        risk_free (float): valor da taxa livre de risco para cálculo do Sharpe.
        plot (bool): se `True`, plota um gráfico de linha com o Sharpe ao longo do tempo.

    Returns:
        pd.Series: uma série com os valores do Sharpe para os últimos `window` dias.
        A série não possui os `window` primeiros dias.

    """
    rolling_sharpe = pd.Series([sharpe_ratio(returns[i - window:i], risk_free)
                                for i in range(window, len(returns))], returns[window:].index)
    if plot:
        fig = px.line(rolling_sharpe, title="Sharpe móvel")
        overall_sharpe = sharpe_ratio(returns, risk_free)
        fig.update_layout(shapes=[
            dict(
                type='line',
                xref='paper', x0=0, x1=1,
                yref='y', y0=overall_sharpe, y1=overall_sharpe,
                line=dict(
                    color='grey',
                    width=2,
                    dash='dash'
                )
            )
        ], annotations=[
            dict(
                text='sharpe total: %.3f' % overall_sharpe,
                xref='paper', x=0.05,
                yref='y', y=overall_sharpe,
                xanchor='left'
            )
        ])
        fig.update_layout(showlegend=False)
        fig.update_xaxes(title_text='Tempo')
        fig.update_yaxes(title_text='Sharpe móvel: ' + str(window) + ' períodos')
        fig.show()
    return rolling_sharpe

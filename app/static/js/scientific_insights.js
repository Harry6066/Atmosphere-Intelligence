/**
 * ATMOSPHERE INTELLIGENCE // Module 10: Scientific Insights & Methodological Limitations
 * Station Bias & Seasonal Bias Grouped Interactive Plotly Visualizations
 */

(function () {
  'use strict';

  function getPlotlyTheme() {
    const isDark = document.documentElement.getAttribute('data-theme') !== 'light';
    return {
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent',
      font: {
        family: 'Inter, system-ui, sans-serif',
        color: isDark ? '#94a3b8' : '#475569',
        size: 11
      },
      gridcolor: isDark ? '#1e293b' : '#e2e8f0',
      zerolinecolor: isDark ? '#334155' : '#cbd5e1'
    };
  }

  // ---------------------------------------------------------------------------
  // 1. Station Bias Grouped Bar Chart
  // ---------------------------------------------------------------------------
  async function renderStationBiasChart() {
    const el = document.getElementById('chart-station-bias');
    if (!el || typeof Plotly === 'undefined') return;

    try {
      const res = await fetch('/api/insights/station-bias');
      const json = await res.json();
      if (!json.success || !json.data || !json.data.stations) return;

      const stations = json.data.stations;
      const theme = getPlotlyTheme();

      const names = stations.map(s => s.short_name);
      const ifRates = stations.map(s => s.if_rate_pct);
      const lofRates = stations.map(s => s.lof_rate_pct);
      const conRates = stations.map(s => s.consensus_rate_pct);
      const ifText = stations.map(s => s.if_fraction_str);
      const lofText = stations.map(s => s.lof_fraction_str);
      const conText = stations.map(s => s.consensus_fraction_str);

      const traceIF = {
        x: names,
        y: ifRates,
        name: 'IF Rate (%)',
        type: 'bar',
        marker: { color: '#06b6d4' },
        customdata: ifText,
        hovertemplate: '<b>%{x}</b><br>IF Rate: %{y:.2f}%<br>Fraction: %{customdata}<extra></extra>'
      };

      const traceLOF = {
        x: names,
        y: lofRates,
        name: 'LOF Rate (%)',
        type: 'bar',
        marker: { color: '#f59e0b' },
        customdata: lofText,
        hovertemplate: '<b>%{x}</b><br>LOF Rate: %{y:.2f}%<br>Fraction: %{customdata}<extra></extra>'
      };

      const traceCon = {
        x: names,
        y: conRates,
        name: 'Consensus (%)',
        type: 'bar',
        marker: { color: '#a855f7' },
        customdata: conText,
        hovertemplate: '<b>%{x}</b><br>Consensus Rate: %{y:.2f}%<br>Fraction: %{customdata}<extra></extra>'
      };

      const layout = {
        paper_bgcolor: theme.paper_bgcolor,
        plot_bgcolor: theme.plot_bgcolor,
        font: theme.font,
        barmode: 'group',
        margin: { l: 40, r: 15, t: 20, b: 65 },
        xaxis: {
          tickangle: -20,
          tickfont: { size: 10 },
          gridcolor: theme.gridcolor
        },
        yaxis: {
          title: 'Detection Rate (%)',
          gridcolor: theme.gridcolor,
          zerolinecolor: theme.zerolinecolor
        },
        legend: {
          orientation: 'h',
          x: 0,
          y: 1.18,
          font: { size: 10 }
        }
      };

      const config = { responsive: true, displayModeBar: false };
      Plotly.newPlot(el, [traceIF, traceLOF, traceCon], layout, config);
    } catch (err) {
      console.error('Failed to render station bias chart:', err);
    }
  }

  // ---------------------------------------------------------------------------
  // 2. Seasonal Bias Grouped Bar Chart
  // ---------------------------------------------------------------------------
  async function renderSeasonalBiasChart() {
    const el = document.getElementById('chart-seasonal-bias');
    if (!el || typeof Plotly === 'undefined') return;

    try {
      const res = await fetch('/api/insights/seasonal-bias');
      const json = await res.json();
      if (!json.success || !json.data || !json.data.seasons) return;

      const seasons = json.data.seasons;
      const theme = getPlotlyTheme();

      const names = seasons.map(s => s.season_name);
      const ifRates = seasons.map(s => s.if_rate_pct);
      const lofRates = seasons.map(s => s.lof_rate_pct);
      const conRates = seasons.map(s => s.consensus_rate_pct);
      const ifText = seasons.map(s => s.if_fraction_str);
      const lofText = seasons.map(s => s.lof_fraction_str);
      const conText = seasons.map(s => s.consensus_fraction_str);

      const traceIF = {
        x: names,
        y: ifRates,
        name: 'IF Rate (%)',
        type: 'bar',
        marker: { color: '#06b6d4' },
        customdata: ifText,
        hovertemplate: '<b>%{x}</b><br>IF Rate: %{y:.2f}%<br>Fraction: %{customdata}<extra></extra>'
      };

      const traceLOF = {
        x: names,
        y: lofRates,
        name: 'LOF Rate (%)',
        type: 'bar',
        marker: { color: '#f59e0b' },
        customdata: lofText,
        hovertemplate: '<b>%{x}</b><br>LOF Rate: %{y:.2f}%<br>Fraction: %{customdata}<extra></extra>'
      };

      const traceCon = {
        x: names,
        y: conRates,
        name: 'Consensus (%)',
        type: 'bar',
        marker: { color: '#a855f7' },
        customdata: conText,
        hovertemplate: '<b>%{x}</b><br>Consensus Rate: %{y:.2f}%<br>Fraction: %{customdata}<extra></extra>'
      };

      const layout = {
        paper_bgcolor: theme.paper_bgcolor,
        plot_bgcolor: theme.plot_bgcolor,
        font: theme.font,
        barmode: 'group',
        margin: { l: 40, r: 15, t: 20, b: 35 },
        xaxis: {
          tickfont: { size: 10 },
          gridcolor: theme.gridcolor
        },
        yaxis: {
          title: 'Detection Rate (%)',
          gridcolor: theme.gridcolor,
          zerolinecolor: theme.zerolinecolor
        },
        legend: {
          orientation: 'h',
          x: 0,
          y: 1.18,
          font: { size: 10 }
        }
      };

      const config = { responsive: true, displayModeBar: false };
      Plotly.newPlot(el, [traceIF, traceLOF, traceCon], layout, config);
    } catch (err) {
      console.error('Failed to render seasonal bias chart:', err);
    }
  }

  function init() {
    renderStationBiasChart();
    renderSeasonalBiasChart();

    const observer = new MutationObserver(() => {
      renderStationBiasChart();
      renderSeasonalBiasChart();
    });
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();


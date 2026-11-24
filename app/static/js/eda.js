/**
 * Atmosphere Intelligence — Exploratory Data Analysis Controller
 * Part of INT396 Unsupervised Machine Learning Project (SIH26073).
 * Renders interactive Plotly.js distribution histograms, box plots,
 * annual climatology curves, and Pearson correlation heatmaps.
 */

const AtmosphereEDA = (function () {
  'use strict';

  let currentStation = 'ALL';
  let currentFeature = 'TMAX';
  let currentSeasonalVar = 'TMAX';
  let currentBins = 30;

  function init() {
    bindEvents();
    if (window.ATMOSPHERE_EDA_INITIAL) {
      renderDistribution(window.ATMOSPHERE_EDA_INITIAL.distribution);
      renderHeatmap(window.ATMOSPHERE_EDA_INITIAL.correlation);
      renderMonthlyClimatology(window.ATMOSPHERE_EDA_INITIAL.seasonal);
    } else {
      refreshDistribution();
      refreshCorrelation();
      refreshSeasonal();
    }
  }

  function bindEvents() {
    const stnSelect = document.getElementById('eda-station-filter');
    const featSelect = document.getElementById('eda-feature-filter');
    const seasSelect = document.getElementById('eda-seasonal-var');
    const binsSelect = document.getElementById('eda-bins-filter');

    if (stnSelect) {
      stnSelect.addEventListener('change', () => {
        currentStation = stnSelect.value;
        refreshDistribution();
        refreshCorrelation();
        refreshSeasonal();
      });
    }

    if (featSelect) {
      featSelect.addEventListener('change', () => {
        currentFeature = featSelect.value;
        refreshDistribution();
      });
    }

    if (seasSelect) {
      seasSelect.addEventListener('change', () => {
        currentSeasonalVar = seasSelect.value;
        refreshSeasonal();
      });
    }

    if (binsSelect) {
      binsSelect.addEventListener('change', () => {
        currentBins = parseInt(binsSelect.value, 10) || 30;
        refreshDistribution();
      });
    }
  }

  // ---------------------------------------------------------------------------
  // 1. Distribution Histogram & Box Plot
  // ---------------------------------------------------------------------------
  function refreshDistribution() {
    const params = new URLSearchParams();
    if (currentStation !== 'ALL') params.set('station', currentStation);
    params.set('feature', currentFeature);
    params.set('bins', currentBins);

    fetch(`/api/eda/distributions?${params.toString()}`)
      .then(res => res.json())
      .then(json => {
        if (json.success) {
          renderDistribution(json.data);
        }
      })
      .catch(err => console.error('Error fetching distributions:', err));
  }

  function renderDistribution(dist) {
    if (!dist || !window.Plotly) return;

    const isDark = document.documentElement.getAttribute('data-theme') !== 'light';
    const textColor = isDark ? '#94a3b8' : '#475569';
    const gridColor = isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)';

    // Update Text Titles & Badges
    const titleEl = document.getElementById('dist-chart-title');
    const countBadge = document.getElementById('dist-count-badge');
    const outliersEl = document.getElementById('dist-outliers-count');
    const skewEl = document.getElementById('dist-skewness');
    const kurtEl = document.getElementById('dist-kurtosis');

    if (titleEl) titleEl.textContent = `${dist.feature_label} Distribution`;
    if (countBadge) countBadge.textContent = `${dist.count.toLocaleString()} points`;
    if (outliersEl) outliersEl.textContent = dist.box.outlier_count.toLocaleString();
    if (skewEl) skewEl.textContent = dist.summary.skewness;
    if (kurtEl) kurtEl.textContent = dist.summary.kurtosis;

    // Update Quantile Summary Cards
    const meanStdEl = document.getElementById('stat-mean-std');
    const medianEl = document.getElementById('stat-median');
    const iqrEl = document.getElementById('stat-iqr');
    const minMaxEl = document.getElementById('stat-min-max');

    if (meanStdEl) meanStdEl.textContent = `${dist.summary.mean} ± ${dist.summary.std}`;
    if (medianEl) medianEl.textContent = `${dist.summary.median}`;
    if (iqrEl) iqrEl.textContent = `${dist.summary.iqr}`;
    if (minMaxEl) minMaxEl.textContent = `${dist.summary.min} / ${dist.summary.max}`;

    // Render Histogram Bar Chart
    const histData = [{
      x: dist.histogram.bin_centers,
      y: dist.histogram.counts,
      type: 'bar',
      marker: {
        color: isDark ? 'rgba(0, 212, 255, 0.7)' : 'rgba(14, 165, 233, 0.8)',
        line: {
          color: isDark ? '#00d4ff' : '#0284c7',
          width: 1
        }
      },
      hoverinfo: 'x+y',
      hovertemplate: `${dist.feature}: %{x:.2f}<br>Count: %{y}<extra></extra>`
    }];

    const histLayout = {
      margin: { t: 20, b: 40, l: 45, r: 20 },
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      font: { family: 'Inter, sans-serif', color: textColor, size: 11 },
      xaxis: {
        title: dist.feature_label,
        gridcolor: gridColor,
        zerolinecolor: gridColor
      },
      yaxis: {
        title: 'Observations',
        gridcolor: gridColor,
        zerolinecolor: gridColor
      }
    };

    Plotly.newPlot('eda-histogram-chart', histData, histLayout, { responsive: true, displayModeBar: false });

    // Render Horizontal Box Plot
    const boxData = [{
      type: 'box',
      q1: [dist.box.q1],
      median: [dist.box.median],
      q3: [dist.box.q3],
      lowerfence: [dist.box.lower_fence],
      upperfence: [dist.box.upper_fence],
      mean: [dist.summary.mean],
      boxpoints: false,
      orientation: 'h',
      marker: { color: '#818cf8' },
      line: { color: isDark ? '#a5b4fc' : '#6366f1' },
      fillcolor: isDark ? 'rgba(129, 140, 248, 0.3)' : 'rgba(99, 102, 241, 0.2)'
    }];

    const boxLayout = {
      margin: { t: 10, b: 25, l: 45, r: 20 },
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      font: { family: 'Inter, sans-serif', color: textColor, size: 10 },
      xaxis: {
        gridcolor: gridColor,
        zerolinecolor: gridColor
      },
      yaxis: {
        showticklabels: false
      }
    };

    Plotly.newPlot('eda-box-chart', boxData, boxLayout, { responsive: true, displayModeBar: false });
  }

  // ---------------------------------------------------------------------------
  // 2. Correlation Matrix Heatmap
  // ---------------------------------------------------------------------------
  function refreshCorrelation() {
    const params = new URLSearchParams();
    if (currentStation !== 'ALL') params.set('station', currentStation);

    fetch(`/api/eda/correlations?${params.toString()}`)
      .then(res => res.json())
      .then(json => {
        if (json.success) renderHeatmap(json.data);
      })
      .catch(err => console.error('Error fetching correlations:', err));
  }

  function renderHeatmap(corr) {
    if (!corr || !window.Plotly) return;

    const isDark = document.documentElement.getAttribute('data-theme') !== 'light';
    const textColor = isDark ? '#94a3b8' : '#475569';

    const zValues = corr.matrix;
    const xLabels = corr.features;
    const yLabels = corr.features;

    const heatmapData = [{
      z: zValues,
      x: xLabels,
      y: yLabels,
      type: 'heatmap',
      colorscale: [
        [0.0, '#3b82f6'],
        [0.5, isDark ? '#0f172a' : '#f8fafc'],
        [1.0, '#ef4444']
      ],
      zmin: -1.0,
      zmax: 1.0,
      showscale: true,
      colorbar: {
        thickness: 12,
        len: 0.9,
        tickfont: { size: 10, color: textColor }
      },
      hoverongaps: false
    }];

    const heatmapLayout = {
      margin: { t: 20, b: 60, l: 80, r: 20 },
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      font: { family: 'JetBrains Mono, monospace', color: textColor, size: 10 },
      xaxis: { tickangle: -45 },
      yaxis: { autorange: 'reversed' }
    };

    Plotly.newPlot('eda-corr-heatmap', heatmapData, heatmapLayout, { responsive: true, displayModeBar: false });
  }

  // ---------------------------------------------------------------------------
  // 3. Monthly Climatology Trajectories
  // ---------------------------------------------------------------------------
  function refreshSeasonal() {
    const params = new URLSearchParams();
    if (currentStation !== 'ALL') params.set('station', currentStation);
    params.set('variable', currentSeasonalVar);

    fetch(`/api/eda/seasonal?${params.toString()}`)
      .then(res => res.json())
      .then(json => {
        if (json.success) renderMonthlyClimatology(json.data);
      })
      .catch(err => console.error('Error fetching seasonal climatology:', err));
  }

  function renderMonthlyClimatology(seasonal) {
    if (!seasonal || !window.Plotly) return;

    const isDark = document.documentElement.getAttribute('data-theme') !== 'light';
    const textColor = isDark ? '#94a3b8' : '#475569';
    const gridColor = isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)';

    const monthLabels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    const stationCurves = seasonal.station_comparison_monthly || {};

    const stationColors = {
      'USW00094728': '#00d4ff', // Central Park Cyan
      'USW00094846': '#818cf8', // Chicago Indigo
      'USW00024233': '#38bdf8', // Seattle Sky
      'USW00012839': '#f59e0b'  // Miami Amber
    };

    const traces = [];

    // If specific station is picked, render its mean with shaded min/max bounds
    if (seasonal.station !== 'All Stations' && seasonal.monthly) {
      const means = seasonal.monthly.map(m => m.mean);
      const mins = seasonal.monthly.map(m => m.min);
      const maxs = seasonal.monthly.map(m => m.max);

      traces.push({
        x: monthLabels,
        y: maxs,
        type: 'scatter',
        mode: 'lines',
        line: { width: 0 },
        showlegend: false,
        hoverinfo: 'none'
      });

      traces.push({
        x: monthLabels,
        y: mins,
        type: 'scatter',
        mode: 'lines',
        fill: 'tonexty',
        fillcolor: isDark ? 'rgba(0, 212, 255, 0.15)' : 'rgba(14, 165, 233, 0.15)',
        line: { width: 0 },
        name: 'Min–Max Envelope'
      });

      traces.push({
        x: monthLabels,
        y: means,
        type: 'scatter',
        mode: 'lines+markers',
        line: { color: '#00d4ff', width: 3 },
        marker: { size: 6, color: '#00d4ff' },
        name: `${seasonal.station} Mean`
      });
    } else {
      // Render all 4 stations simultaneously for comparative view
      for (const [stnId, curve] of Object.entries(stationCurves)) {
        const color = stationColors[stnId] || '#38bdf8';
        const shortName = curve.name.split(',')[0].replace(' INTERNATIONAL AIRPORT', ' INTL').replace(' OHARE', '');

        traces.push({
          x: monthLabels,
          y: curve.values,
          type: 'scatter',
          mode: 'lines+markers',
          line: { color: color, width: 2.5 },
          marker: { size: 5, color: color },
          name: `${shortName} (${stnId})`
        });
      }
    }

    const layout = {
      margin: { t: 20, b: 40, l: 45, r: 20 },
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      font: { family: 'Inter, sans-serif', color: textColor, size: 11 },
      xaxis: { gridcolor: gridColor },
      yaxis: {
        title: `${seasonal.variable} Climatology`,
        gridcolor: gridColor
      },
      legend: {
        orientation: 'h',
        y: 1.15,
        x: 0,
        font: { size: 10, color: textColor }
      }
    };

    Plotly.newPlot('eda-monthly-chart', traces, layout, { responsive: true, displayModeBar: false });
  }

  return {
    init: init
  };
})();

document.addEventListener('DOMContentLoaded', AtmosphereEDA.init);


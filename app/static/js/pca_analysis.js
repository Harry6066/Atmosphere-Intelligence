/**
 * ATMOSPHERE INTELLIGENCE // Module 08: PCA Dimensionality Reduction
 * Interactive 2D & 3D Eigenspace Projections, Scree Eigenvalue Spectrum, and Loadings Heatmap
 */

(function () {
  'use strict';

  // State management
  const state = {
    viewMode: '2d', // '2d' or '3d'
    colorBy: 'consensus', // 'consensus', 'station', 'season'
    station: '',
    season: '',
    sample: true,
    isDark: document.documentElement.getAttribute('data-theme') !== 'light'
  };

  // Plotly chart theme helper
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
  // 1. Fetch & Render PCA Coordinates (2D or 3D)
  // ---------------------------------------------------------------------------
  async function renderPcaProjection() {
    const el2d = document.getElementById('chart-pca-2d');
    const el3d = document.getElementById('chart-pca-3d');
    if (typeof Plotly === 'undefined') return;

    const params = new URLSearchParams({
      sample: state.sample ? 'true' : 'false',
      color_by: state.colorBy
    });
    if (state.station) params.append('station', state.station);
    if (state.season) params.append('season', state.season);

    try {
      const res = await fetch(`/api/pca/coordinates?${params.toString()}`);
      const json = await res.json();
      if (!json.success) return;

      const points = json.data;
      const theme = getPlotlyTheme();

      // Group points by color category
      const groups = {};
      points.forEach(p => {
        let cat = p.consensus;
        if (state.colorBy === 'station') cat = p.station;
        else if (state.colorBy === 'season') cat = p.season;

        if (!groups[cat]) {
          groups[cat] = {
            x: [],
            y: [],
            z: [],
            text: [],
            customdata: []
          };
        }

        const hover = `<b>${p.station}</b> &bull; ${p.date}<br>` +
          `Status: <b>${p.consensus}</b> &bull; Season: ${p.season}<br>` +
          `PC1: ${p.pc1.toFixed(3)} &bull; PC2: ${p.pc2.toFixed(3)} &bull; PC3: ${p.pc3.toFixed(3)}<br>` +
          `TMAX: ${p.tmax != null ? p.tmax.toFixed(1) : '--'}&deg;C &bull; TMIN: ${p.tmin != null ? p.tmin.toFixed(1) : '--'}&deg;C &bull; PRCP: ${p.prcp != null ? p.prcp.toFixed(1) : '--'}mm`;

        groups[cat].x.push(p.pc1);
        groups[cat].y.push(p.pc2);
        groups[cat].z.push(p.pc3);
        groups[cat].text.push(hover);
        groups[cat].customdata.push([p.station_id, p.date]);
      });

      // Color maps
      const paletteConsensus = {
        'Consensus': '#ef4444',
        'IF_Only': '#f59e0b',
        'LOF_Only': '#a855f7',
        'Normal': state.isDark ? '#475569' : '#94a3b8'
      };

      const paletteStation = {
        'Chicago O\'Hare': '#38bdf8',
        'Miami International': '#f43f5e',
        'New York JFK': '#a855f7',
        'Seattle-Tacoma': '#10b981'
      };

      const paletteSeason = {
        'Winter': '#38bdf8',
        'Spring': '#4ade80',
        'Summer': '#f59e0b',
        'Fall': '#fb923c'
      };

      function getColor(name) {
        if (state.colorBy === 'station') return paletteStation[name] || '#38bdf8';
        if (state.colorBy === 'season') return paletteSeason[name] || '#4ade80';
        return paletteConsensus[name] || (state.isDark ? '#475569' : '#94a3b8');
      }

      function getSize(name) {
        if (state.colorBy === 'consensus') {
          if (name === 'Consensus') return 7;
          if (name === 'IF_Only' || name === 'LOF_Only') return 6;
          return 4;
        }
        return 5;
      }

      function getOpacity(name) {
        if (state.colorBy === 'consensus' && name === 'Normal') return 0.35;
        return 0.85;
      }

      // -----------------------------------------------------------------------
      // Render 2D Projection
      // -----------------------------------------------------------------------
      if (state.viewMode === '2d' && el2d) {
        const traces2d = Object.keys(groups).map(k => ({
          name: k === 'Consensus' ? 'Consensus (Both)' : k,
          x: groups[k].x,
          y: groups[k].y,
          text: groups[k].text,
          customdata: groups[k].customdata,
          mode: 'markers',
          type: 'scatter',
          hoverinfo: 'text',
          marker: {
            color: getColor(k),
            size: getSize(k),
            opacity: getOpacity(k),
            line: (k === 'Consensus') ? { color: '#ffffff', width: 1 } : undefined
          }
        }));

        const layout2d = {
          paper_bgcolor: theme.paper_bgcolor,
          plot_bgcolor: theme.plot_bgcolor,
          font: theme.font,
          margin: { l: 50, r: 20, t: 20, b: 50 },
          legend: { orientation: 'h', y: 1.08, x: 0 },
          xaxis: {
            title: 'PC1: Thermal & Annual Cycle (27.96% variance)',
            gridcolor: theme.gridcolor,
            zerolinecolor: theme.zerolinecolor
          },
          yaxis: {
            title: 'PC2: Synoptic Frontal Volatility (16.08% variance)',
            gridcolor: theme.gridcolor,
            zerolinecolor: theme.zerolinecolor
          }
        };

        Plotly.newPlot(el2d, traces2d, layout2d, { responsive: true });

        el2d.on('plotly_click', function (data) {
          if (!data || !data.points || !data.points[0]) return;
          const pt = data.points[0];
          if (pt.customdata && pt.customdata[0] && pt.customdata[1]) {
            if (typeof window.openObservationModal === 'function') {
              window.openObservationModal(pt.customdata[0], pt.customdata[1]);
            }
          }
        });
      }

      // -----------------------------------------------------------------------
      // Render 3D Projection
      // -----------------------------------------------------------------------
      if (state.viewMode === '3d' && el3d) {
        const traces3d = Object.keys(groups).map(k => ({
          name: k === 'Consensus' ? 'Consensus (Both)' : k,
          x: groups[k].x,
          y: groups[k].y,
          z: groups[k].z,
          text: groups[k].text,
          customdata: groups[k].customdata,
          mode: 'markers',
          type: 'scatter3d',
          hoverinfo: 'text',
          marker: {
            color: getColor(k),
            size: getSize(k) + 1,
            opacity: getOpacity(k)
          }
        }));

        const layout3d = {
          paper_bgcolor: theme.paper_bgcolor,
          plot_bgcolor: theme.plot_bgcolor,
          font: theme.font,
          margin: { l: 0, r: 0, t: 0, b: 0 },
          scene: {
            xaxis: { title: 'PC1 (Thermal)', gridcolor: theme.gridcolor },
            yaxis: { title: 'PC2 (Volatility)', gridcolor: theme.gridcolor },
            zaxis: { title: 'PC3 (Moisture)', gridcolor: theme.gridcolor },
            camera: {
              eye: { x: 1.5, y: 1.5, z: 1.2 }
            }
          },
          legend: { orientation: 'h', y: 0.98, x: 0.05 }
        };

        Plotly.newPlot(el3d, traces3d, layout3d, { responsive: true });

        el3d.on('plotly_click', function (data) {
          if (!data || !data.points || !data.points[0]) return;
          const pt = data.points[0];
          if (pt.customdata && pt.customdata[0] && pt.customdata[1]) {
            if (typeof window.openObservationModal === 'function') {
              window.openObservationModal(pt.customdata[0], pt.customdata[1]);
            }
          }
        });
      }

    } catch (err) {
      console.error('Error rendering PCA projection:', err);
    }
  }

  // ---------------------------------------------------------------------------
  // 2. Render Scree Plot (Eigenvalue Spectrum)
  // ---------------------------------------------------------------------------
  async function renderScreePlot() {
    const el = document.getElementById('chart-pca-scree');
    if (!el || typeof Plotly === 'undefined') return;

    try {
      const res = await fetch('/api/pca/scree');
      const json = await res.json();
      if (!json.success) return;

      const items = json.data.components;
      const comps = items.map(c => `PC${c.component}`);
      const indVars = items.map(c => c.explained_variance_pct);
      const cumVars = items.map(c => c.cumulative_variance_pct);

      const theme = getPlotlyTheme();

      const traceBars = {
        name: 'Individual Variance',
        x: comps,
        y: indVars,
        type: 'bar',
        marker: { color: '#06b6d4' }
      };

      const traceLine = {
        name: 'Cumulative Variance',
        x: comps,
        y: cumVars,
        type: 'scatter',
        mode: 'lines+markers',
        yaxis: 'y2',
        line: { color: '#a855f7', width: 2.5 },
        marker: { color: '#c084fc', size: 6 }
      };

      const layout = {
        paper_bgcolor: theme.paper_bgcolor,
        plot_bgcolor: theme.plot_bgcolor,
        font: theme.font,
        margin: { l: 45, r: 45, t: 30, b: 40 },
        legend: { orientation: 'h', y: 1.15, x: 0 },
        xaxis: { gridcolor: theme.gridcolor },
        yaxis: {
          title: 'Individual %',
          gridcolor: theme.gridcolor,
          zerolinecolor: theme.zerolinecolor
        },
        yaxis2: {
          title: 'Cumulative %',
          overlaying: 'y',
          side: 'right',
          range: [0, 105],
          showgrid: false
        },
        shapes: [
          // 80% threshold line
          {
            type: 'line',
            xref: 'paper',
            x0: 0,
            x1: 1,
            yref: 'y2',
            y0: 80,
            y1: 80,
            line: { color: 'rgba(239, 68, 68, 0.4)', width: 1.5, dash: 'dash' }
          }
        ]
      };

      Plotly.newPlot(el, [traceBars, traceLine], layout, { responsive: true, displayModeBar: false });
    } catch (err) {
      console.error('Error rendering Scree plot:', err);
    }
  }

  // ---------------------------------------------------------------------------
  // 3. Render Eigenvector Loadings Heatmap
  // ---------------------------------------------------------------------------
  async function renderLoadingsHeatmap() {
    const el = document.getElementById('chart-pca-loadings');
    if (!el || typeof Plotly === 'undefined') return;

    try {
      const res = await fetch('/api/pca/loadings');
      const json = await res.json();
      if (!json.success) return;

      const items = json.data.loadings;
      const features = items.map(d => d.feature).reverse();
      const pc1 = items.map(d => d.pc1).reverse();
      const pc2 = items.map(d => d.pc2).reverse();
      const pc3 = items.map(d => d.pc3).reverse();

      const zValues = features.map((_, i) => [pc1[i], pc2[i], pc3[i]]);
      const theme = getPlotlyTheme();

      const data = [{
        z: zValues,
        x: ['PC1 (Thermal)', 'PC2 (Volatility)', 'PC3 (Moisture)'],
        y: features,
        type: 'heatmap',
        colorscale: [
          [0.0, '#3b82f6'],
          [0.5, state.isDark ? '#0f172a' : '#f8fafc'],
          [1.0, '#ef4444']
        ],
        zmin: -0.7,
        zmax: 0.7,
        hoverongaps: false
      }];

      // Format numerical text annotations
      const annotations = [];
      for (let i = 0; i < features.length; i++) {
        for (let j = 0; j < 3; j++) {
          const val = zValues[i][j];
          annotations.push({
            x: j,
            y: features[i],
            text: (val >= 0 ? '+' : '') + val.toFixed(3),
            font: {
              family: 'JetBrains Mono, monospace',
              size: 10,
              color: Math.abs(val) > 0.4 ? '#ffffff' : (state.isDark ? '#94a3b8' : '#334155')
            },
            showarrow: false
          });
        }
      }

      const layout = {
        paper_bgcolor: theme.paper_bgcolor,
        plot_bgcolor: theme.plot_bgcolor,
        font: theme.font,
        margin: { l: 150, r: 30, t: 30, b: 40 },
        annotations: annotations,
        xaxis: { side: 'bottom', gridcolor: theme.gridcolor },
        yaxis: { automargin: true, gridcolor: theme.gridcolor }
      };

      Plotly.newPlot(el, data, layout, { responsive: true, displayModeBar: false });
    } catch (err) {
      console.error('Error rendering loadings heatmap:', err);
    }
  }

  // ---------------------------------------------------------------------------
  // 4. View Mode & Filter Controls Setup
  // ---------------------------------------------------------------------------
  function setupControls() {
    const btn2d = document.getElementById('btn-view-2d');
    const btn3d = document.getElementById('btn-view-3d');
    const c2d = document.getElementById('pca-2d-container');
    const c3d = document.getElementById('pca-3d-container');

    if (btn2d && btn3d && c2d && c3d) {
      btn2d.addEventListener('click', () => {
        state.viewMode = '2d';
        btn2d.className = 'px-3 py-1 rounded font-mono font-bold text-xs bg-cyan-500/20 text-cyan-300 transition-colors';
        btn3d.className = 'px-3 py-1 rounded font-mono text-xs text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors';
        c2d.classList.remove('hidden');
        c3d.classList.add('hidden');
        renderPcaProjection();
      });

      btn3d.addEventListener('click', () => {
        state.viewMode = '3d';
        btn3d.className = 'px-3 py-1 rounded font-mono font-bold text-xs bg-purple-500/20 text-purple-300 transition-colors';
        btn2d.className = 'px-3 py-1 rounded font-mono text-xs text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors';
        c3d.classList.remove('hidden');
        c2d.classList.add('hidden');
        renderPcaProjection();
      });
    }

    // Color By selector
    const colorSelect = document.getElementById('pca-color-by');
    if (colorSelect) {
      colorSelect.addEventListener('change', e => {
        state.colorBy = e.target.value;
        renderPcaProjection();
      });
    }

    // Station Filter
    const stationFilter = document.getElementById('pca-station-filter');
    if (stationFilter) {
      stationFilter.addEventListener('change', e => {
        state.station = e.target.value;
        renderPcaProjection();
      });
    }

    // Season Filter
    const seasonFilter = document.getElementById('pca-season-filter');
    if (seasonFilter) {
      seasonFilter.addEventListener('change', e => {
        state.season = e.target.value;
        renderPcaProjection();
      });
    }

    // Sample Toggle
    const sampleToggle = document.getElementById('pca-sample-toggle');
    if (sampleToggle) {
      sampleToggle.addEventListener('change', e => {
        state.sample = e.target.checked;
        renderPcaProjection();
      });
    }

    // Theme Mutation Observer
    const observer = new MutationObserver(mutations => {
      mutations.forEach(m => {
        if (m.attributeName === 'data-theme') {
          state.isDark = document.documentElement.getAttribute('data-theme') !== 'light';
          renderPcaProjection();
          renderScreePlot();
          renderLoadingsHeatmap();
        }
      });
    });
    observer.observe(document.documentElement, { attributes: true });
  }

  // DOM Loaded
  document.addEventListener('DOMContentLoaded', () => {
    setupControls();
    renderPcaProjection();
    renderScreePlot();
    renderLoadingsHeatmap();
  });

})();


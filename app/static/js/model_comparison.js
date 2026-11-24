/**
 * ATMOSPHERE INTELLIGENCE // Module 07: Model Comparison Engine
 * Interactive Plotly visualizations, Bivariate Decision Quadrants, and Observation Ledger
 */

(function () {
  'use strict';

  // Global state
  const state = {
    currentPage: 1,
    perPage: 15,
    totalPages: 1,
    station: '',
    season: '',
    year: '',
    temporalMode: 'seasonal', // 'seasonal' or 'yearly'
    status: 'Consensus', // Default view focuses on high-confidence consensus
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
  // 1. Venn / Overlap Donut Chart
  // ---------------------------------------------------------------------------
  // 1. Set-Theoretic Partition Four-Segment Stacked Bar
  // ---------------------------------------------------------------------------
  function renderVennDonut() {
    const el = document.getElementById('chart-venn-donut');
    if (!el || typeof Plotly === 'undefined') return;

    const theme = getPlotlyTheme();

    const traceNeither = {
      y: ['Full Dataset'],
      x: [11064],
      name: 'Neither (Normal)',
      type: 'bar',
      orientation: 'h',
      marker: { color: state.isDark ? '#334155' : '#94a3b8' },
      hoverinfo: 'name+x+text',
      text: ['11,064 (94.66%)'],
      textposition: 'inside',
      textfont: { color: '#ffffff', size: 10 }
    };

    const traceIF = {
      y: ['Full Dataset', 'Anomaly Union'],
      x: [273, 273],
      name: 'IF Only',
      type: 'bar',
      orientation: 'h',
      marker: { color: '#06b6d4' },
      hoverinfo: 'name+x+text',
      text: ['273 (2.34%)', '273 (43.75%)'],
      textposition: 'inside',
      textfont: { color: '#ffffff', size: 10 }
    };

    const traceLOF = {
      y: ['Full Dataset', 'Anomaly Union'],
      x: [273, 273],
      name: 'LOF Only',
      type: 'bar',
      orientation: 'h',
      marker: { color: '#f59e0b' },
      hoverinfo: 'name+x+text',
      text: ['273 (2.34%)', '273 (43.75%)'],
      textposition: 'inside',
      textfont: { color: '#000000', size: 10 }
    };

    const traceConsensus = {
      y: ['Full Dataset', 'Anomaly Union'],
      x: [78, 78],
      name: 'Consensus (Both)',
      type: 'bar',
      orientation: 'h',
      marker: { color: '#ef4444' },
      hoverinfo: 'name+x+text',
      text: ['78 (0.67%)', '78 (12.50%)'],
      textposition: 'inside',
      textfont: { color: '#ffffff', size: 10 }
    };

    const layout = {
      paper_bgcolor: theme.paper_bgcolor,
      plot_bgcolor: theme.plot_bgcolor,
      font: theme.font,
      barmode: 'stack',
      autosize: true,
      margin: { l: 95, r: 15, t: 25, b: 35, pad: 2 },
      xaxis: {
        title: { text: 'Observation Count', font: { size: 10 } },
        gridcolor: theme.gridcolor,
        zerolinecolor: theme.zerolinecolor
      },
      yaxis: {
        gridcolor: theme.gridcolor,
        tickfont: { size: 11, family: 'JetBrains Mono, monospace' }
      },
      legend: {
        orientation: 'h',
        x: 0,
        y: 1.25,
        font: { size: 10 }
      }
    };

    Plotly.react(el, [traceNeither, traceIF, traceLOF, traceConsensus], layout, { responsive: true, displayModeBar: false });
  }

  // ---------------------------------------------------------------------------
  // 2. Station Comparison Bar Chart
  // ---------------------------------------------------------------------------
  async function renderStationComparison() {
    const el = document.getElementById('chart-station-comparison');
    if (!el || typeof Plotly === 'undefined') return;

    try {
      const res = await fetch('/api/comparison/station');
      const json = await res.json();
      if (!json.success) return;

      const items = json.data;
      const stations = items.map(d => d.name);
      const ifVals = items.map(d => d.if_anomalies);
      const lofVals = items.map(d => d.lof_anomalies);
      const consVals = items.map(d => d.consensus_anomalies);

      const theme = getPlotlyTheme();
      const traces = [
        {
          name: 'Isolation Forest',
          x: stations,
          y: ifVals,
          type: 'bar',
          marker: { color: '#f59e0b' }
        },
        {
          name: 'Local Outlier Factor',
          x: stations,
          y: lofVals,
          type: 'bar',
          marker: { color: '#a855f7' }
        },
        {
          name: 'Consensus (Both)',
          x: stations,
          y: consVals,
          type: 'bar',
          marker: { color: '#ef4444' }
        }
      ];

      const layout = {
        barmode: 'group',
        paper_bgcolor: theme.paper_bgcolor,
        plot_bgcolor: theme.plot_bgcolor,
        font: theme.font,
        margin: { l: 40, r: 20, t: 30, b: 40 },
        legend: { orientation: 'h', y: 1.15, x: 0 },
        xaxis: { gridcolor: theme.gridcolor },
        yaxis: { title: 'Anomalies', gridcolor: theme.gridcolor, zerolinecolor: theme.zerolinecolor }
      };

      Plotly.newPlot(el, traces, layout, { responsive: true, displayModeBar: false });
    } catch (err) {
      console.error('Error rendering station comparison:', err);
    }
  }

  // ---------------------------------------------------------------------------
  // 3. Seasonal Comparison Bar Chart
  // ---------------------------------------------------------------------------
  async function renderSeasonalComparison() {
    const el = document.getElementById('chart-seasonal-comparison');
    if (!el || typeof Plotly === 'undefined') return;

    try {
      const res = await fetch('/api/comparison/seasonal');
      const json = await res.json();
      if (!json.success) return;

      const items = json.data;
      const seasons = items.map(d => d.season);
      const ifVals = items.map(d => d.if_anomalies);
      const lofVals = items.map(d => d.lof_anomalies);
      const consVals = items.map(d => d.consensus_anomalies);

      const theme = getPlotlyTheme();
      const traces = [
        {
          name: 'Isolation Forest',
          x: seasons,
          y: ifVals,
          type: 'bar',
          marker: { color: '#f59e0b' }
        },
        {
          name: 'Local Outlier Factor',
          x: seasons,
          y: lofVals,
          type: 'bar',
          marker: { color: '#a855f7' }
        },
        {
          name: 'Consensus (Both)',
          x: seasons,
          y: consVals,
          type: 'bar',
          marker: { color: '#ef4444' }
        }
      ];

      const layout = {
        barmode: 'group',
        paper_bgcolor: theme.paper_bgcolor,
        plot_bgcolor: theme.plot_bgcolor,
        font: theme.font,
        margin: { l: 40, r: 20, t: 30, b: 40 },
        legend: { orientation: 'h', y: 1.15, x: 0 },
        xaxis: { gridcolor: theme.gridcolor },
        yaxis: { title: 'Anomalies', gridcolor: theme.gridcolor, zerolinecolor: theme.zerolinecolor }
      };

      Plotly.newPlot(el, traces, layout, { responsive: true, displayModeBar: false });
    } catch (err) {
      console.error('Error rendering seasonal comparison:', err);
    }
  }

  // ---------------------------------------------------------------------------
  // 3b. Yearly Comparison Bar Chart
  // ---------------------------------------------------------------------------
  async function renderYearlyComparison() {
    const el = document.getElementById('chart-seasonal-comparison');
    if (!el || typeof Plotly === 'undefined') return;

    try {
      const res = await fetch('/api/comparison/yearly');
      const json = await res.json();
      if (!json.success) return;

      const items = json.data;
      const years = items.map(d => String(d.year));
      const ifVals = items.map(d => d.if_anomalies);
      const lofVals = items.map(d => d.lof_anomalies);
      const consVals = items.map(d => d.consensus_anomalies);

      const theme = getPlotlyTheme();
      const traces = [
        {
          name: 'Isolation Forest',
          x: years,
          y: ifVals,
          type: 'bar',
          marker: { color: '#f59e0b' }
        },
        {
          name: 'Local Outlier Factor',
          x: years,
          y: lofVals,
          type: 'bar',
          marker: { color: '#a855f7' }
        },
        {
          name: 'Consensus (Both)',
          x: years,
          y: consVals,
          type: 'bar',
          marker: { color: '#ef4444' }
        }
      ];

      const layout = {
        barmode: 'group',
        paper_bgcolor: theme.paper_bgcolor,
        plot_bgcolor: theme.plot_bgcolor,
        font: theme.font,
        margin: { l: 40, r: 20, t: 30, b: 40 },
        legend: { orientation: 'h', y: 1.15, x: 0 },
        xaxis: { title: 'Observation Year', gridcolor: theme.gridcolor },
        yaxis: { title: 'Anomalies', gridcolor: theme.gridcolor, zerolinecolor: theme.zerolinecolor }
      };

      Plotly.newPlot(el, traces, layout, { responsive: true, displayModeBar: false });
    } catch (err) {
      console.error('Error rendering yearly comparison:', err);
    }
  }

  // ---------------------------------------------------------------------------
  // 4. Bivariate Score Scatter (IF Score vs LOF Score Quadrant)
  // ---------------------------------------------------------------------------
  async function renderBivariateScatter() {
    const el = document.getElementById('chart-score-scatter');
    if (!el || typeof Plotly === 'undefined') return;

    try {
      const res = await fetch('/api/comparison/scatter?n_points=1200');
      const json = await res.json();
      if (!json.success) return;

      const points = json.data;
      const theme = getPlotlyTheme();

      // Separate points into categories for visual layering
      const groups = {
        normal: { x: [], y: [], text: [], customdata: [] },
        if_only: { x: [], y: [], text: [], customdata: [] },
        lof_only: { x: [], y: [], text: [], customdata: [] },
        consensus: { x: [], y: [], text: [], customdata: [] }
      };

      points.forEach(p => {
        const hover = `<b>${p.station}</b> &bull; ${p.date}<br>` +
          `Status: <b>${p.consensus}</b><br>` +
          `IF Score: ${p.if_score} (Anomaly: ${p.if_anomaly})<br>` +
          `LOF Score: ${p.lof_score} (Anomaly: ${p.lof_anomaly})<br>` +
          `TMAX: ${p.tmax}&deg;C &bull; TMIN: ${p.tmin}&deg;C &bull; PRCP: ${p.prcp}mm`;

        let grp = groups.normal;
        if (p.consensus === 'Consensus') grp = groups.consensus;
        else if (p.consensus === 'IF_Only') grp = groups.if_only;
        else if (p.consensus === 'LOF_Only') grp = groups.lof_only;

        grp.x.push(p.if_score);
        grp.y.push(p.lof_score);
        grp.text.push(hover);
        grp.customdata.push([p.station_id, p.date]);
      });

      const traces = [
        {
          name: 'Normal (Both)',
          x: groups.normal.x,
          y: groups.normal.y,
          text: groups.normal.text,
          customdata: groups.normal.customdata,
          mode: 'markers',
          type: 'scatter',
          hoverinfo: 'text',
          marker: {
            color: state.isDark ? '#334155' : '#cbd5e1',
            size: 4,
            opacity: 0.35
          }
        },
        {
          name: 'IF Only',
          x: groups.if_only.x,
          y: groups.if_only.y,
          text: groups.if_only.text,
          customdata: groups.if_only.customdata,
          mode: 'markers',
          type: 'scatter',
          hoverinfo: 'text',
          marker: {
            color: '#f59e0b',
            size: 6,
            opacity: 0.85
          }
        },
        {
          name: 'LOF Only',
          x: groups.lof_only.x,
          y: groups.lof_only.y,
          text: groups.lof_only.text,
          customdata: groups.lof_only.customdata,
          mode: 'markers',
          type: 'scatter',
          hoverinfo: 'text',
          marker: {
            color: '#a855f7',
            size: 6,
            opacity: 0.85
          }
        },
        {
          name: 'Consensus (Both)',
          x: groups.consensus.x,
          y: groups.consensus.y,
          text: groups.consensus.text,
          customdata: groups.consensus.customdata,
          mode: 'markers',
          type: 'scatter',
          hoverinfo: 'text',
          marker: {
            color: '#ef4444',
            size: 7,
            opacity: 0.95,
            line: { color: '#ffffff', width: 1 }
          }
        }
      ];

      const layout = {
        paper_bgcolor: theme.paper_bgcolor,
        plot_bgcolor: theme.plot_bgcolor,
        font: theme.font,
        margin: { l: 50, r: 20, t: 30, b: 50 },
        legend: { orientation: 'h', y: 1.12, x: 0 },
        xaxis: {
          title: 'Isolation Forest Score (Negative = Anomalous)',
          gridcolor: theme.gridcolor,
          zerolinecolor: theme.zerolinecolor
        },
        yaxis: {
          title: 'LOF Score (Negative Outlier Factor)',
          gridcolor: theme.gridcolor,
          zerolinecolor: theme.zerolinecolor
        }
      };

      Plotly.newPlot(el, traces, layout, { responsive: true });

      // Click to inspect
      el.on('plotly_click', function (data) {
        if (!data || !data.points || !data.points[0]) return;
        const pt = data.points[0];
        if (pt.customdata && pt.customdata[0] && pt.customdata[1]) {
          if (typeof window.openObservationModal === 'function') {
            window.openObservationModal(pt.customdata[0], pt.customdata[1]);
          }
        }
      });
    } catch (err) {
      console.error('Error rendering bivariate scatter:', err);
    }
  }

  // ---------------------------------------------------------------------------
  // 5. Comparison Observation Ledger (Table & Pagination)
  // ---------------------------------------------------------------------------
  async function loadObservations() {
    const tbody = document.getElementById('comparison-table-body');
    const summaryInfo = document.getElementById('table-summary-info');
    const pageIndicator = document.getElementById('page-indicator');
    const prevBtn = document.getElementById('btn-prev-page');
    const nextBtn = document.getElementById('btn-next-page');

    if (!tbody) return;

    tbody.innerHTML = `<tr><td colspan="12" class="py-8 text-center text-xs text-[var(--text-muted)]">Loading comparison records...</td></tr>`;

    const params = new URLSearchParams({
      page: state.currentPage,
      per_page: state.perPage
    });
    if (state.station) params.append('station', state.station);
    if (state.season) params.append('season', state.season);
    if (state.year) params.append('year', state.year);
    if (state.status) params.append('status', state.status);

    try {
      const res = await fetch(`/api/comparison/observations?${params.toString()}`);
      const json = await res.json();
      if (!json.success) {
        tbody.innerHTML = `<tr><td colspan="12" class="py-6 text-center text-xs text-red-400">Failed to load comparison records.</td></tr>`;
        return;
      }

      const d = json.data;
      state.totalPages = d.pages || 1;

      // Update Summary Info
      if (summaryInfo) {
        summaryInfo.textContent = `Showing ${d.records.length > 0 ? (d.page - 1) * d.per_page + 1 : 0} to ${Math.min(d.page * d.per_page, d.total)} of ${d.total.toLocaleString()} observations`;
      }
      if (pageIndicator) {
        pageIndicator.textContent = `Page ${d.page} of ${d.pages || 1}`;
      }
      if (prevBtn) prevBtn.disabled = d.page <= 1;
      if (nextBtn) nextBtn.disabled = d.page >= d.pages;

      if (d.records.length === 0) {
        tbody.innerHTML = `<tr><td colspan="12" class="py-8 text-center text-xs text-[var(--text-muted)]">No observations match the selected criteria.</td></tr>`;
        return;
      }

      // Render Rows
      tbody.innerHTML = d.records.map(row => {
        // Status Badges
        let statusBadge = '<span class="px-2 py-0.5 rounded text-[10px] bg-slate-500/20 text-[var(--text-muted)]">Normal</span>';
        if (row.consensus === 'Consensus') {
          statusBadge = '<span class="px-2 py-0.5 rounded text-[10px] bg-red-500/20 text-red-400 font-bold border border-red-500/30">CONSENSUS</span>';
        } else if (row.consensus === 'IF_Only') {
          statusBadge = '<span class="px-2 py-0.5 rounded text-[10px] bg-amber-500/20 text-amber-300 border border-amber-500/30">IF ONLY</span>';
        } else if (row.consensus === 'LOF_Only') {
          statusBadge = '<span class="px-2 py-0.5 rounded text-[10px] bg-purple-500/20 text-purple-300 border border-purple-500/30">LOF ONLY</span>';
        }

        const ifBadge = row.if_anomaly
          ? '<span class="px-1.5 py-0.5 rounded text-[10px] bg-amber-500/20 text-amber-400 font-bold">ANOMALY</span>'
          : '<span class="text-[10px] text-[var(--text-muted)]">Normal</span>';

        const lofBadge = row.lof_anomaly
          ? '<span class="px-1.5 py-0.5 rounded text-[10px] bg-purple-500/20 text-purple-400 font-bold">ANOMALY</span>'
          : '<span class="text-[10px] text-[var(--text-muted)]">Normal</span>';

        return `
          <tr class="hover:bg-[var(--bg-card-hover)] transition-colors">
            <td class="py-2.5 px-3 font-semibold text-[var(--text-primary)]">${row.station_name}</td>
            <td class="py-2.5 px-3 text-[var(--text-muted)]">${row.date}</td>
            <td class="py-2.5 px-3 text-right ${row.tmax != null ? (row.tmax > 30 ? 'text-rose-400' : (row.tmax < 0 ? 'text-cyan-300' : 'text-[var(--text-primary)]')) : 'text-[var(--text-muted)]'}">
              ${row.tmax != null ? row.tmax.toFixed(1) + '&deg;C' : '--'}
            </td>
            <td class="py-2.5 px-3 text-right ${row.tmin != null ? (row.tmin < -10 ? 'text-cyan-400' : 'text-[var(--text-primary)]') : 'text-[var(--text-muted)]'}">
              ${row.tmin != null ? row.tmin.toFixed(1) + '&deg;C' : '--'}
            </td>
            <td class="py-2.5 px-3 text-right ${row.prcp > 10 ? 'text-sky-400 font-bold' : 'text-[var(--text-muted)]'}">
              ${row.prcp != null ? row.prcp.toFixed(1) + 'mm' : '--'}
            </td>
            <td class="py-2.5 px-3 text-right ${row.snow > 0 ? 'text-indigo-300 font-bold' : 'text-[var(--text-muted)]'}">
              ${row.snow != null ? row.snow.toFixed(1) + 'mm' : '--'}
            </td>
            <td class="py-2.5 px-3 text-center">${ifBadge}</td>
            <td class="py-2.5 px-3 text-center">${lofBadge}</td>
            <td class="py-2.5 px-3 text-center">${statusBadge}</td>
            <td class="py-2.5 px-3 text-right font-mono text-[11px] ${row.if_score < 0 ? 'text-amber-400 font-bold' : 'text-[var(--text-muted)]'}">
              ${row.if_score != null ? row.if_score.toFixed(4) : '--'}
            </td>
            <td class="py-2.5 px-3 text-right font-mono text-[11px] ${row.lof_score < -1.5 ? 'text-purple-400 font-bold' : 'text-[var(--text-muted)]'}">
              ${row.lof_score != null ? row.lof_score.toFixed(4) : '--'}
            </td>
            <td class="py-2.5 px-3 text-center">
              <button 
                onclick="window.openObservationModal('${row.station_id}', '${row.date}')"
                class="p-1 rounded text-cyan-400 hover:text-cyan-300 hover:bg-cyan-500/10 transition-colors"
                title="Inspect Observation">
                <i data-lucide="scan" class="w-3.5 h-3.5 inline"></i>
              </button>
            </td>
          </tr>
        `;
      }).join('');

      if (window.lucide && typeof window.lucide.createIcons === 'function') {
        window.lucide.createIcons();
      }
    } catch (err) {
      console.error('Error fetching comparison observations:', err);
      tbody.innerHTML = `<tr><td colspan="12" class="py-6 text-center text-xs text-red-400">Error retrieving data.</td></tr>`;
    }
  }

  // ---------------------------------------------------------------------------
  // 6. Event Handlers & Initialization
  // ---------------------------------------------------------------------------
  function setupListeners() {
    // Status filter
    const statusSelect = document.getElementById('filter-status');
    if (statusSelect) {
      statusSelect.addEventListener('change', e => {
        state.status = e.target.value;
        state.currentPage = 1;
        loadObservations();
      });
    }

    // Station filter
    const stationSelect = document.getElementById('filter-station');
    if (stationSelect) {
      stationSelect.addEventListener('change', e => {
        state.station = e.target.value;
        state.currentPage = 1;
        loadObservations();
      });
    }

    // Season filter
    const seasonSelect = document.getElementById('filter-season');
    if (seasonSelect) {
      seasonSelect.addEventListener('change', e => {
        state.season = e.target.value;
        state.currentPage = 1;
        loadObservations();
      });
    }

    // Year filter
    const yearSelect = document.getElementById('filter-year');
    if (yearSelect) {
      yearSelect.addEventListener('change', e => {
        state.year = e.target.value;
        state.currentPage = 1;
        loadObservations();
      });
    }

    // Pagination buttons
    const prevBtn = document.getElementById('btn-prev-page');
    if (prevBtn) {
      prevBtn.addEventListener('click', () => {
        if (state.currentPage > 1) {
          state.currentPage--;
          loadObservations();
        }
      });
    }

    const nextBtn = document.getElementById('btn-next-page');
    if (nextBtn) {
      nextBtn.addEventListener('click', () => {
        if (state.currentPage < state.totalPages) {
          state.currentPage++;
          loadObservations();
        }
      });
    }

    // Temporal breakdown toggle (Seasonal vs Yearly)
    const btnSeasonal = document.getElementById('btn-toggle-seasonal');
    const btnYearly = document.getElementById('btn-toggle-yearly');
    const titleTemporal = document.getElementById('temporal-chart-title');

    if (btnSeasonal && btnYearly) {
      btnSeasonal.addEventListener('click', () => {
        state.temporalMode = 'seasonal';
        btnSeasonal.className = 'px-2.5 py-0.5 rounded font-bold text-cyan-300 bg-cyan-500/20 transition-colors';
        btnYearly.className = 'px-2.5 py-0.5 rounded text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors';
        if (titleTemporal) titleTemporal.textContent = 'Seasonal Anomaly Concentration Comparison';
        renderSeasonalComparison();
      });

      btnYearly.addEventListener('click', () => {
        state.temporalMode = 'yearly';
        btnYearly.className = 'px-2.5 py-0.5 rounded font-bold text-cyan-300 bg-cyan-500/20 transition-colors';
        btnSeasonal.className = 'px-2.5 py-0.5 rounded text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors';
        if (titleTemporal) titleTemporal.textContent = 'Annual Anomaly Trajectory Comparison (2016–2023)';
        renderYearlyComparison();
      });
    }

    // Theme change observer
    const observer = new MutationObserver(mutations => {
      mutations.forEach(m => {
        if (m.attributeName === 'data-theme') {
          state.isDark = document.documentElement.getAttribute('data-theme') !== 'light';
          renderVennDonut();
          renderStationComparison();
          if (state.temporalMode === 'yearly') {
            renderYearlyComparison();
          } else {
            renderSeasonalComparison();
          }
          renderBivariateScatter();
        }
      });
    });
    observer.observe(document.documentElement, { attributes: true });
  }

  // DOM Content Loaded
  document.addEventListener('DOMContentLoaded', () => {
    renderVennDonut();
    renderStationComparison();
    renderSeasonalComparison();
    renderBivariateScatter();
    loadObservations();
    setupListeners();
  });

})();


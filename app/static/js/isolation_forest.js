/**
 * ATMOSPHERE INTELLIGENCE — Module 5: Isolation Forest Controller
 * Client-side Plotly rendering, dynamic model inference, and observation inspection.
 */

document.addEventListener('DOMContentLoaded', () => {
  initRadioCards();
  if (window.ATMOSPHERE_IF_INITIAL) {
    renderAllIFCharts(window.ATMOSPHERE_IF_INITIAL);
  }
  if (typeof lucide !== 'undefined') {
    lucide.createIcons();
  }
});

function initRadioCards() {
  document.querySelectorAll('.atm-radio-card').forEach(card => {
    card.addEventListener('click', () => {
      const parent = card.closest('div');
      parent.querySelectorAll('.atm-radio-card').forEach(c => {
        c.classList.remove('border-cyan-400', 'bg-cyan-500/10', 'text-cyan-300', 'font-bold', 'active');
        c.classList.add('border-[var(--border-subtle)]', 'bg-[var(--bg-card)]');
      });
      card.classList.remove('border-[var(--border-subtle)]', 'bg-[var(--bg-card)]');
      card.classList.add('border-cyan-400', 'bg-cyan-500/10', 'text-cyan-300', 'font-bold', 'active');
      const radio = card.querySelector('input[type="radio"]');
      if (radio) radio.checked = true;
    });
  });
}

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
    gridcolor: isDark ? 'rgba(51, 65, 85, 0.4)' : 'rgba(226, 232, 240, 0.8)',
    linecolor: isDark ? 'rgba(51, 65, 85, 0.7)' : 'rgba(203, 213, 225, 1)'
  };
}

function renderAllIFCharts(data) {
  renderScoreHistogram(data.histogram, data.summary.score_threshold);
  renderStationBar(data.station_distribution);
  renderSeasonBar(data.season_distribution);
  renderTempScatter(data.scatter_sample);
}

function renderScoreHistogram(hist, threshold) {
  const chartEl = document.getElementById('if-score-hist-chart');
  if (!chartEl || !hist) return;
  const theme = getPlotlyTheme();

  // Split bins into normal and anomaly around threshold = 0.0
  const colors = hist.bin_centers.map(c => c < threshold ? '#f43f5e' : '#00d4ff');

  const trace = {
    x: hist.bin_centers,
    y: hist.frequencies,
    type: 'bar',
    marker: {
      color: colors,
      line: { color: 'rgba(0,0,0,0.2)', width: 1 }
    },
    name: 'Observations',
    hovertemplate: 'Score Bin: %{x:.4f}<br>Count: %{y}<extra></extra>'
  };

  const layout = {
    margin: { l: 45, r: 20, t: 25, b: 40 },
    paper_bgcolor: theme.paper_bgcolor,
    plot_bgcolor: theme.plot_bgcolor,
    font: theme.font,
    xaxis: {
      title: 'Continuous Decision Score (decision_function)',
      gridcolor: theme.gridcolor,
      linecolor: theme.linecolor,
      zeroline: true,
      zerolinecolor: '#f43f5e',
      zerolinewidth: 2
    },
    yaxis: {
      title: 'Frequency',
      gridcolor: theme.gridcolor,
      linecolor: theme.linecolor
    },
    shapes: [{
      type: 'line',
      x0: threshold,
      x1: threshold,
      y0: 0,
      y1: Math.max(...hist.frequencies) * 1.05,
      line: {
        color: '#f43f5e',
        width: 2,
        dash: 'dash'
      }
    }],
    annotations: [{
      x: threshold,
      y: Math.max(...hist.frequencies) * 0.95,
      text: 'Decision Boundary (score = 0.0)',
      showarrow: true,
      arrowhead: 2,
      ax: -60,
      ay: -30,
      font: { color: '#f43f5e', size: 10 }
    }]
  };

  Plotly.react(chartEl, [trace], layout, { responsive: true, displayModeBar: false });
}

function renderStationBar(stDist) {
  const chartEl = document.getElementById('if-station-bar-chart');
  if (!chartEl || !stDist) return;
  const theme = getPlotlyTheme();

  const names = stDist.map(s => s.name.split(',')[0]);
  const anomalies = stDist.map(s => s.anomalies);
  const rates = stDist.map(s => s.rate_pct);

  const trace = {
    x: names,
    y: anomalies,
    type: 'bar',
    marker: {
      color: ['#00d4ff', '#38bdf8', '#818cf8', '#a78bfa'],
      line: { color: 'rgba(0,0,0,0.2)', width: 1 }
    },
    text: rates.map(r => `${r}% rate`),
    textposition: 'outside',
    hovertemplate: '%{x}<br>Anomalies: %{y}<br>Rate: %{text}<extra></extra>'
  };

  const layout = {
    margin: { l: 45, r: 20, t: 25, b: 60 },
    paper_bgcolor: theme.paper_bgcolor,
    plot_bgcolor: theme.plot_bgcolor,
    font: theme.font,
    xaxis: {
      gridcolor: theme.gridcolor,
      linecolor: theme.linecolor
    },
    yaxis: {
      title: 'Flagged Anomalies',
      gridcolor: theme.gridcolor,
      linecolor: theme.linecolor
    }
  };

  Plotly.react(chartEl, [trace], layout, { responsive: true, displayModeBar: false });
}

function renderSeasonBar(seasonDist) {
  const chartEl = document.getElementById('if-season-bar-chart');
  if (!chartEl || !seasonDist) return;
  const theme = getPlotlyTheme();

  const seasons = seasonDist.map(s => s.season);
  const counts = seasonDist.map(s => s.count);
  const pcts = seasonDist.map(s => s.pct);

  const trace = {
    x: seasons,
    y: counts,
    type: 'bar',
    marker: {
      color: ['#38bdf8', '#34d399', '#fbbf24', '#f97316'],
      line: { color: 'rgba(0,0,0,0.2)', width: 1 }
    },
    text: pcts.map(p => `${p}%`),
    textposition: 'outside',
    hovertemplate: '%{x}<br>Count: %{y} (%{text})<extra></extra>'
  };

  const layout = {
    margin: { l: 45, r: 20, t: 25, b: 40 },
    paper_bgcolor: theme.paper_bgcolor,
    plot_bgcolor: theme.plot_bgcolor,
    font: theme.font,
    xaxis: {
      gridcolor: theme.gridcolor,
      linecolor: theme.linecolor
    },
    yaxis: {
      title: 'Anomaly Count',
      gridcolor: theme.gridcolor,
      linecolor: theme.linecolor
    }
  };

  Plotly.react(chartEl, [trace], layout, { responsive: true, displayModeBar: false });
}

function renderTempScatter(sample) {
  const chartEl = document.getElementById('if-temp-scatter-chart');
  if (!chartEl || !sample) return;
  const theme = getPlotlyTheme();

  const normals = sample.filter(p => p.is_anomaly === 0);
  const anomalies = sample.filter(p => p.is_anomaly === 1);

  const traceNorm = {
    x: normals.map(p => p.tmin),
    y: normals.map(p => p.tmax),
    mode: 'markers',
    type: 'scatter',
    name: 'Normal Weather',
    marker: {
      color: 'rgba(0, 212, 255, 0.35)',
      size: 5
    },
    hovertemplate: 'TMIN: %{x}&deg;C<br>TMAX: %{y}&deg;C<extra>Normal</extra>'
  };

  const traceAnom = {
    x: anomalies.map(p => p.tmin),
    y: anomalies.map(p => p.tmax),
    mode: 'markers',
    type: 'scatter',
    name: 'Isolation Anomaly',
    marker: {
      color: '#f43f5e',
      size: 8,
      symbol: 'x',
      line: { color: '#f43f5e', width: 2 }
    },
    hovertemplate: 'TMIN: %{x}&deg;C<br>TMAX: %{y}&deg;C<extra>IF Outlier</extra>'
  };

  const layout = {
    margin: { l: 45, r: 20, t: 25, b: 40 },
    paper_bgcolor: theme.paper_bgcolor,
    plot_bgcolor: theme.plot_bgcolor,
    font: theme.font,
    showlegend: true,
    legend: { x: 0.02, y: 0.98, bgcolor: 'rgba(0,0,0,0.4)' },
    xaxis: {
      title: 'Minimum Temperature (&deg;C)',
      gridcolor: theme.gridcolor,
      linecolor: theme.linecolor
    },
    yaxis: {
      title: 'Maximum Temperature (&deg;C)',
      gridcolor: theme.gridcolor,
      linecolor: theme.linecolor
    }
  };

  Plotly.react(chartEl, [traceNorm, traceAnom], layout, { responsive: true, displayModeBar: false });
}

async function executeIsolationForest() {
  const form = document.getElementById('if-config-form');
  const runBtn = document.getElementById('run-if-btn');
  const statusBadge = document.getElementById('execution-badge');
  const statusText = document.getElementById('execution-status-text');

  // Collect parameters
  const contaminationRadio = form.querySelector('input[name="contamination"]:checked');
  const contamination = contaminationRadio ? parseFloat(contaminationRadio.value) : 0.03;
  const nEstimators = parseInt(document.getElementById('n-estimators-select').value, 10);
  const randomState = parseInt(document.getElementById('random-seed-input').value, 10);
  const stationScope = document.getElementById('station-scope-select').value;

  const featureCheckboxes = form.querySelectorAll('input[name="features"]:checked');
  const features = Array.from(featureCheckboxes).map(cb => cb.value);

  if (features.length === 0) {
    alert('Please select at least one modeling feature.');
    return;
  }

  // Set loading state
  runBtn.disabled = true;
  runBtn.innerHTML = '<span class="inline-block animate-spin mr-2">&#9696;</span> Executing Model...';
  statusText.innerText = 'Training Isolation Forest...';

  try {
    const isBaselineParams = (
      (contamination === 0.01 || contamination === 0.03 || contamination === 0.05) &&
      nEstimators === 100 &&
      features.length === 10
    );

    let data;
    if (isBaselineParams) {
      const resp = await fetch(`/api/isolation-forest/results?contamination=${contamination}&n_estimators=${nEstimators}&station=${stationScope}`);
      const json = await resp.json();
      if (!json.success) throw new Error(json.error);
      data = json.data;
      statusText.innerText = 'Baseline Cache Loaded (0 ms)';
    } else {
      const resp = await fetch('/api/isolation-forest/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          contamination: contamination,
          n_estimators: nEstimators,
          random_state: randomState,
          features: features
        })
      });
      const json = await resp.json();
      if (!json.success) throw new Error(json.error);
      data = json.data;
      statusText.innerText = `Live Inference Complete (${data.execution.duration_ms} ms, ${data.execution.cached ? 'from cache' : 'freshly trained'})`;
    }

    updateIFDashboard(data);
  } catch (err) {
    console.error('Model execution error:', err);
    alert(`Execution failed: ${err.message}`);
    statusText.innerText = 'Execution Failed';
  } finally {
    runBtn.disabled = false;
    runBtn.innerHTML = '<i data-lucide="play" class="w-4 h-4"></i><span>Execute Model</span>';
    if (typeof lucide !== 'undefined') lucide.createIcons();
  }
}

function updateIFDashboard(data) {
  // Update KPI Telemetry
  const summary = data.summary || data.results;
  if (summary) {
    document.getElementById('stat-total-obs').innerText = (summary.total_observations || 11688).toLocaleString();
    document.getElementById('stat-anom-count').innerText = summary.anomalies_detected;
    document.getElementById('stat-anom-pct').innerText = `${summary.anomaly_percentage}%`;
    document.getElementById('stat-norm-count').innerText = (summary.normal_observations || 0).toLocaleString();
    document.getElementById('stat-norm-pct').innerText = `${(100 - summary.anomaly_percentage).toFixed(2)}%`;
    document.getElementById('stat-score-range').innerText = `[${summary.score_min.toFixed(4)}, ${summary.score_max.toFixed(4)}]`;
    document.getElementById('stat-score-median').innerText = summary.score_median.toFixed(4);
  }

  // Update Charts
  if (data.histogram) renderScoreHistogram(data.histogram, 0.0);
  if (data.station_distribution) renderStationBar(data.station_distribution);
  if (data.season_distribution) renderSeasonBar(data.season_distribution);
  if (data.scatter_sample) renderTempScatter(data.scatter_sample);

  // Update Top Suspicious Table if present
  if (data.top_anomalies) {
    const tbody = document.getElementById('if-anomalies-tbody');
    tbody.innerHTML = '';
    data.top_anomalies.slice(0, 20).forEach((anom, idx) => {
      const tr = document.createElement('tr');
      tr.className = 'hover:bg-[var(--bg-hover)] transition-colors';
      tr.innerHTML = `
        <td class="p-2.5 font-bold text-amber-400">#${anom.rank || idx + 1}</td>
        <td class="p-2.5 text-[var(--text-primary)] font-bold">${anom.date}</td>
        <td class="p-2.5 text-[var(--text-secondary)] font-sans">
          <span class="font-mono text-cyan-400 font-semibold">${anom.station}</span>
          <span class="text-[11px] block text-[var(--text-muted)]">${anom.station_name || anom.station}</span>
        </td>
        <td class="p-2.5 text-right">${anom.tmax.toFixed(1)}</td>
        <td class="p-2.5 text-right">${anom.tmin.toFixed(1)}</td>
        <td class="p-2.5 text-right text-cyan-400">${(anom.temp_range || anom.tmax - anom.tmin).toFixed(1)}</td>
        <td class="p-2.5 text-right ${anom.prcp > 25.0 ? 'text-amber-400 font-bold' : ''}">${anom.prcp.toFixed(1)}</td>
        <td class="p-2.5 text-right ${anom.snow > 0.0 ? 'text-sky-300 font-bold' : ''}">${(anom.snow || 0).toFixed(1)}</td>
        <td class="p-2.5 text-right font-bold text-rose-400">${anom.score.toFixed(4)}</td>
        <td class="p-2.5 text-center">
          <button onclick="inspectObservation('${anom.station}', '${anom.date}')" class="atm-btn-secondary px-2.5 py-1 text-[11px]">
            Inspect
          </button>
        </td>
      `;
      tbody.appendChild(tr);
    });
  }
}

function resetIFBaseline() {
  const form = document.getElementById('if-config-form');
  // Select c = 0.03
  const c03Radio = form.querySelector('input[name="contamination"][value="0.03"]');
  if (c03Radio) {
    c03Radio.click();
  }
  document.getElementById('n-estimators-select').value = '100';
  document.getElementById('station-scope-select').value = 'ALL';
  form.querySelectorAll('input[name="features"]').forEach(cb => { cb.checked = true; });
  executeIsolationForest();
}

async function inspectObservation(station, date) {
  try {
    const resp = await fetch(`/api/observation/inspect?station=${station}&date=${date}`);
    const json = await resp.json();
    if (!json.success) throw new Error(json.error);
    const ob = json.data;

    // Populate Modal
    document.getElementById('modal-status-badge').innerText = ob.models.consensus.status_label;
    document.getElementById('modal-station-id').innerText = ob.station;
    document.getElementById('modal-date').innerText = `${ob.date} (${ob.season})`;
    document.getElementById('modal-station-name').innerText = ob.station_name;
    document.getElementById('modal-station-meta').innerText = `${ob.climate_zone} // Elev: ${ob.coordinates.elevation_m}m // Lat: ${ob.coordinates.latitude}, Lon: ${ob.coordinates.longitude}`;

    document.getElementById('modal-tmax').innerText = `${ob.observations.tmax.toFixed(1)} °C`;
    document.getElementById('modal-tmin').innerText = `${ob.observations.tmin.toFixed(1)} °C`;
    document.getElementById('modal-trange').innerText = `${ob.observations.temp_range.toFixed(1)} °C`;
    document.getElementById('modal-prcp').innerText = `${ob.observations.prcp.toFixed(1)} mm`;
    document.getElementById('modal-snow').innerText = `${ob.observations.snow.toFixed(1)} mm`;
    document.getElementById('modal-snwd').innerText = `${ob.observations.snwd.toFixed(1)} mm`;
    document.getElementById('modal-tmax-diff').innerText = `${ob.observations.tmax_diff_1d.toFixed(1)} °C`;
    document.getElementById('modal-tmin-diff').innerText = `${ob.observations.tmin_diff_1d.toFixed(1)} °C`;

    // IF Verdict
    document.getElementById('modal-if-verdict').innerText = ob.models.isolation_forest.prediction;
    document.getElementById('modal-if-score').innerText = ob.models.isolation_forest.score.toFixed(4);
    document.getElementById('modal-if-desc').innerText = ob.models.isolation_forest.interpretation;

    // LOF Verdict
    document.getElementById('modal-lof-verdict').innerText = ob.models.local_outlier_factor.prediction;
    document.getElementById('modal-lof-score').innerText = ob.models.local_outlier_factor.negative_outlier_factor.toFixed(4);
    document.getElementById('modal-lof-offset').innerText = ob.models.local_outlier_factor.offset_threshold.toFixed(4);
    document.getElementById('modal-lof-desc').innerText = ob.models.local_outlier_factor.interpretation;

    // Scaled grid
    const scaledGrid = document.getElementById('modal-scaled-grid');
    scaledGrid.innerHTML = '';
    for (const [feat, val] of Object.entries(ob.scaled_features)) {
      const isExtreme = Math.abs(val) > 2.5;
      const card = document.createElement('div');
      card.className = `p-2 rounded border ${isExtreme ? 'border-amber-500/40 bg-amber-500/10 text-amber-300' : 'border-[var(--border-subtle)] bg-[var(--bg-card)]'}`;
      card.innerHTML = `
        <div class="text-[9px] text-[var(--text-muted)] truncate">${feat}</div>
        <div class="font-bold font-mono mt-0.5">${val > 0 ? '+' : ''}${val.toFixed(2)} &sigma;</div>
      `;
      scaledGrid.appendChild(card);
    }

    // NOAA flags
    document.getElementById('modal-qc-status').innerText = ob.quality_flags.prcp_qflag === 'Pass' ? 'PASSED (0 Failed QFLAGs)' : ob.quality_flags.prcp_qflag;
    document.getElementById('modal-trace-flag').innerText = ob.quality_flags.has_trace_precipitation ? 'Yes (Trace accumulation < 0.1mm)' : 'None';

    // Show modal
    document.getElementById('observation-modal').classList.remove('hidden');
    if (typeof lucide !== 'undefined') lucide.createIcons();
  } catch (err) {
    alert(`Could not load observation details: ${err.message}`);
  }
}

function closeObservationModal() {
  const modal = document.getElementById('observation-modal');
  if (modal) modal.classList.add('hidden');
}

// Close modal on escape key
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') closeObservationModal();
});


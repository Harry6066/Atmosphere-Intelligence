/**
 * ATMOSPHERE INTELLIGENCE — Module 6: Local Outlier Factor (LOF) Controller
 * Client-side Plotly rendering, dynamic k-NN density inference, and observation inspection.
 */

document.addEventListener('DOMContentLoaded', () => {
  initLOFRadioCards();
  if (window.ATMOSPHERE_LOF_INITIAL) {
    renderAllLOFCharts(window.ATMOSPHERE_LOF_INITIAL);
  }
  if (typeof lucide !== 'undefined') {
    lucide.createIcons();
  }
});

function initLOFRadioCards() {
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

function renderAllLOFCharts(data) {
  renderLOFScoreHistogram(data.histogram, data.summary.offset_threshold);
  renderLOFStationBar(data.station_distribution);
  renderLOFSeasonBar(data.season_distribution);
  renderLOFTempScatter(data.scatter_sample);
}

function renderLOFScoreHistogram(hist, offset) {
  const chartEl = document.getElementById('lof-score-hist-chart');
  if (!chartEl || !hist) return;
  const theme = getPlotlyTheme();

  // Color bins: below offset = red (anomaly); above offset = cyan (inlier)
  const colors = hist.bin_centers.map(c => c < offset ? '#f43f5e' : '#00d4ff');

  const trace = {
    x: hist.bin_centers,
    y: hist.frequencies,
    type: 'bar',
    marker: {
      color: colors,
      line: { color: 'rgba(0,0,0,0.2)', width: 1 }
    },
    name: 'Observations',
    hovertemplate: 'NOF Bin: %{x:.4f}<br>Count: %{y}<extra></extra>'
  };

  const layout = {
    margin: { l: 45, r: 20, t: 25, b: 40 },
    paper_bgcolor: theme.paper_bgcolor,
    plot_bgcolor: theme.plot_bgcolor,
    font: theme.font,
    xaxis: {
      title: 'Negative Outlier Factor (negative_outlier_factor_)',
      gridcolor: theme.gridcolor,
      linecolor: theme.linecolor
    },
    yaxis: {
      title: 'Frequency',
      gridcolor: theme.gridcolor,
      linecolor: theme.linecolor
    },
    shapes: [{
      type: 'line',
      x0: offset,
      x1: offset,
      y0: 0,
      y1: Math.max(...hist.frequencies) * 1.05,
      line: {
        color: '#f43f5e',
        width: 2,
        dash: 'dash'
      }
    }],
    annotations: [{
      x: offset,
      y: Math.max(...hist.frequencies) * 0.95,
      text: `Offset Threshold (${offset.toFixed(4)})`,
      showarrow: true,
      arrowhead: 2,
      ax: -70,
      ay: -30,
      font: { color: '#f43f5e', size: 10 }
    }]
  };

  Plotly.react(chartEl, [trace], layout, { responsive: true, displayModeBar: false });
}

function renderLOFStationBar(stDist) {
  const chartEl = document.getElementById('lof-station-bar-chart');
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
      color: ['#00d4ff', '#f59e0b', '#38bdf8', '#818cf8'],
      line: { color: 'rgba(0,0,0,0.2)', width: 1 }
    },
    text: rates.map(r => `${r}% rate`),
    textposition: 'outside',
    hovertemplate: '%{x}<br>LOF Anomalies: %{y}<br>Rate: %{text}<extra></extra>'
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
      title: 'Flagged Density Outliers',
      gridcolor: theme.gridcolor,
      linecolor: theme.linecolor
    }
  };

  Plotly.react(chartEl, [trace], layout, { responsive: true, displayModeBar: false });
}

function renderLOFSeasonBar(seasonDist) {
  const chartEl = document.getElementById('lof-season-bar-chart');
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

function renderLOFTempScatter(sample) {
  const chartEl = document.getElementById('lof-temp-scatter-chart');
  if (!chartEl || !sample) return;
  const theme = getPlotlyTheme();

  const normals = sample.filter(p => p.is_anomaly === 0);
  const anomalies = sample.filter(p => p.is_anomaly === 1);

  const traceNorm = {
    x: normals.map(p => p.tmin),
    y: normals.map(p => p.tmax),
    mode: 'markers',
    type: 'scatter',
    name: 'Normal Density',
    marker: {
      color: 'rgba(0, 212, 255, 0.35)',
      size: 5
    },
    hovertemplate: 'TMIN: %{x}&deg;C<br>TMAX: %{y}&deg;C<extra>Normal Density</extra>'
  };

  const traceAnom = {
    x: anomalies.map(p => p.tmin),
    y: anomalies.map(p => p.tmax),
    mode: 'markers',
    type: 'scatter',
    name: 'LOF Density Outlier',
    marker: {
      color: '#f43f5e',
      size: 8,
      symbol: 'diamond',
      line: { color: '#f43f5e', width: 2 }
    },
    hovertemplate: 'TMIN: %{x}&deg;C<br>TMAX: %{y}&deg;C<extra>LOF Outlier</extra>'
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

async function executeLOF() {
  const form = document.getElementById('lof-config-form');
  const runBtn = document.getElementById('run-lof-btn');
  const statusText = document.getElementById('lof-execution-status-text');

  const nNeighborsRadio = form.querySelector('input[name="n_neighbors"]:checked');
  const nNeighbors = nNeighborsRadio ? parseInt(nNeighborsRadio.value, 10) : 20;
  const contamination = parseFloat(document.getElementById('lof-contamination-select').value);
  const metric = document.getElementById('lof-metric-select').value;
  const stationScope = document.getElementById('lof-station-scope-select').value;

  const featureCheckboxes = form.querySelectorAll('input[name="lof_features"]:checked');
  const features = Array.from(featureCheckboxes).map(cb => cb.value);

  if (features.length === 0) {
    alert('Please select at least one modeling feature.');
    return;
  }

  runBtn.disabled = true;
  runBtn.innerHTML = '<span class="inline-block animate-spin mr-2">&#9696;</span> Fitting LOF...';
  statusText.innerText = 'Executing Local Outlier Factor (novelty=False)...';

  try {
    const isBaselineParams = (
      (nNeighbors === 10 || nNeighbors === 20 || nNeighbors === 30) &&
      contamination === 0.03 &&
      metric === 'minkowski' &&
      features.length === 10
    );

    let data;
    if (isBaselineParams) {
      const resp = await fetch(`/api/lof/results?n_neighbors=${nNeighbors}&contamination=${contamination}&station=${stationScope}`);
      const json = await resp.json();
      if (!json.success) throw new Error(json.error);
      data = json.data;
      statusText.innerText = 'Baseline Cache Loaded (0 ms)';
    } else {
      const resp = await fetch('/api/lof/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          n_neighbors: nNeighbors,
          contamination: contamination,
          metric: metric,
          features: features
        })
      });
      const json = await resp.json();
      if (!json.success) throw new Error(json.error);
      data = json.data;
      statusText.innerText = `Live Batch Inference (${data.execution.duration_ms} ms, ${data.execution.cached ? 'from cache' : 'freshly fitted'})`;
    }

    updateLOFDashboard(data);
  } catch (err) {
    console.error('LOF execution error:', err);
    alert(`Execution failed: ${err.message}`);
    statusText.innerText = 'Execution Failed';
  } finally {
    runBtn.disabled = false;
    runBtn.innerHTML = '<i data-lucide="play" class="w-4 h-4"></i><span>Run LOF Model</span>';
    if (typeof lucide !== 'undefined') lucide.createIcons();
  }
}

function updateLOFDashboard(data) {
  const summary = data.summary || data.results;
  if (summary) {
    document.getElementById('lof-stat-total').innerText = (summary.total_observations || 11688).toLocaleString();
    document.getElementById('lof-stat-anom').innerText = summary.anomalies_detected;
    document.getElementById('lof-stat-pct').innerText = `${summary.anomaly_percentage}%`;
    const offset = summary.offset_threshold || summary.decision_offset_threshold;
    if (offset !== undefined) {
      document.getElementById('lof-stat-offset').innerText = offset.toFixed(4);
    }
    const minVal = summary.score_min !== undefined ? summary.score_min : summary.negative_outlier_factor_min;
    const maxVal = summary.score_max !== undefined ? summary.score_max : summary.negative_outlier_factor_max;
    const medVal = summary.score_median !== undefined ? summary.score_median : summary.negative_outlier_factor_median;
    if (minVal !== undefined && maxVal !== undefined) {
      document.getElementById('lof-stat-span').innerText = `[${minVal.toFixed(2)}, ${maxVal.toFixed(2)}]`;
    }
    if (medVal !== undefined) {
      document.getElementById('lof-stat-median').innerText = medVal.toFixed(4);
    }
  }

  const offsetVal = summary ? (summary.offset_threshold || summary.decision_offset_threshold || -1.3632) : -1.3632;
  if (data.histogram) renderLOFScoreHistogram(data.histogram, offsetVal);
  if (data.station_distribution) renderLOFStationBar(data.station_distribution);
  if (data.season_distribution) renderLOFSeasonBar(data.season_distribution);
  if (data.scatter_sample) renderLOFTempScatter(data.scatter_sample);

  if (data.top_anomalies) {
    const tbody = document.getElementById('lof-anomalies-tbody');
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

function resetLOFBaseline() {
  const form = document.getElementById('lof-config-form');
  const k20Radio = form.querySelector('input[name="n_neighbors"][value="20"]');
  if (k20Radio) k20Radio.click();
  document.getElementById('lof-contamination-select').value = '0.03';
  document.getElementById('lof-metric-select').value = 'minkowski';
  document.getElementById('lof-station-scope-select').value = 'ALL';
  form.querySelectorAll('input[name="lof_features"]').forEach(cb => { cb.checked = true; });
  executeLOF();
}

async function inspectObservation(station, date) {
  try {
    const resp = await fetch(`/api/observation/inspect?station=${station}&date=${date}`);
    const json = await resp.json();
    if (!json.success) throw new Error(json.error);
    const ob = json.data;

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

    document.getElementById('modal-if-verdict').innerText = ob.models.isolation_forest.prediction;
    document.getElementById('modal-if-score').innerText = ob.models.isolation_forest.score.toFixed(4);
    document.getElementById('modal-if-desc').innerText = ob.models.isolation_forest.interpretation;

    document.getElementById('modal-lof-verdict').innerText = ob.models.local_outlier_factor.prediction;
    document.getElementById('modal-lof-score').innerText = ob.models.local_outlier_factor.negative_outlier_factor.toFixed(4);
    document.getElementById('modal-lof-offset').innerText = ob.models.local_outlier_factor.offset_threshold.toFixed(4);
    document.getElementById('modal-lof-desc').innerText = ob.models.local_outlier_factor.interpretation;

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

    document.getElementById('modal-qc-status').innerText = ob.quality_flags.prcp_qflag === 'Pass' ? 'PASSED (0 Failed QFLAGs)' : ob.quality_flags.prcp_qflag;
    document.getElementById('modal-trace-flag').innerText = ob.quality_flags.has_trace_precipitation ? 'Yes (Trace accumulation < 0.1mm)' : 'None';

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

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') closeObservationModal();
});


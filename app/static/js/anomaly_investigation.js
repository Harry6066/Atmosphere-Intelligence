/**
 * ATMOSPHERE INTELLIGENCE // Module 09: Anomaly Investigation & Extreme Events
 * Historical Timeline, Observation Inspector, Surrounding Context Window, and Triage Ledger
 */

const AtmosphereInvestigation = (function () {
  'use strict';

  // State Store
  const state = {
    page: 1,
    perPage: 15,
    totalPages: 1,
    station: 'ALL',
    classification: 'consensus',
    year: 'ALL',
    season: 'ALL',
    qflag: 'all',
    search: '',
    sortBy: 'date',
    sortDir: 'desc',
    timelineStation: 'ALL',
    timelineYear: 'ALL',
    currentStation: 'USW00094846',
    currentDate: '2019-01-30',
    isDark: document.documentElement.getAttribute('data-theme') !== 'light'
  };

  // Plotly theme helper
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
  // 1. Timeline Chart (Plotly)
  // ---------------------------------------------------------------------------
  async function loadTimelineChart() {
    const chartEl = document.getElementById('chart-investigation-timeline');
    const loadingEl = document.getElementById('timeline-loading-state');
    const errorEl = document.getElementById('timeline-error-state');
    const emptyEl = document.getElementById('timeline-empty-state');

    if (!chartEl || typeof Plotly === 'undefined') return;

    // Display loading state only while the API request is pending
    if (loadingEl) loadingEl.classList.remove('hidden');
    if (errorEl) errorEl.classList.add('hidden');
    if (emptyEl) emptyEl.classList.add('hidden');

    try {
      const url = new URL('/api/investigation/timeline', window.location.origin);
      if (state.timelineStation !== 'ALL') url.searchParams.set('station', state.timelineStation);
      if (state.timelineYear !== 'ALL') url.searchParams.set('year', state.timelineYear);

      const res = await fetch(url.toString());
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}: ${res.statusText}`);
      }

      const json = await res.json();
      if (!json.success || !json.data) {
        throw new Error(json.error || 'Failed to retrieve timeline data');
      }

      const d = json.data;

      // Handle empty observation state
      if (!d.months || d.months.length === 0 || d.total_obs === 0) {
        if (loadingEl) loadingEl.classList.add('hidden');
        if (emptyEl) emptyEl.classList.remove('hidden');
        chartEl.innerHTML = '';
        return;
      }

      const theme = getPlotlyTheme();
      const isAllYears = state.timelineYear === 'ALL';

      // Smart X-Axis formatting: prevent collision and ensure readable year labels from 2016 through 2023
      let tickvals = undefined;
      let ticktext = undefined;
      if (isAllYears && d.months.length >= 24) {
        tickvals = [];
        ticktext = [];
        d.months.forEach((m) => {
          if (m.endsWith('-01')) {
            tickvals.push(m);
            ticktext.push(m.slice(0, 4)); // "2016", "2017", ..., "2023"
          } else if (m.endsWith('-07') && d.months.length <= 48) {
            tickvals.push(m);
            ticktext.push('Mid-' + m.slice(2, 4));
          }
        });
      }

      const traceIF = {
        x: d.months,
        y: d.if_anomalies,
        name: 'Isolation Forest',
        type: 'bar',
        marker: { color: '#06b6d4' }
      };

      const traceLOF = {
        x: d.months,
        y: d.lof_anomalies,
        name: 'LOF (k=20)',
        type: 'bar',
        marker: { color: '#f59e0b' }
      };

      const traceConsensus = {
        x: d.months,
        y: d.consensus_anomalies,
        name: 'Consensus Both',
        type: 'bar',
        marker: { color: '#a855f7' }
      };

      const layout = {
        paper_bgcolor: theme.paper_bgcolor,
        plot_bgcolor: theme.plot_bgcolor,
        font: theme.font,
        barmode: 'group',
        autosize: true,
        margin: { l: 48, r: 20, t: 36, b: 48, pad: 4 },
        xaxis: {
          gridcolor: theme.gridcolor,
          tickvals: tickvals,
          ticktext: ticktext,
          tickmode: tickvals ? 'array' : 'auto',
          tickangle: isAllYears ? 0 : -30,
          tickfont: { size: 11, family: 'JetBrains Mono, monospace' }
        },
        yaxis: {
          title: {
            text: 'Monthly Anomaly Count',
            font: { size: 11 }
          },
          gridcolor: theme.gridcolor,
          zerolinecolor: theme.zerolinecolor
        },
        legend: {
          orientation: 'h',
          x: 0.5,
          xanchor: 'center',
          y: 1.12,
          yanchor: 'bottom',
          font: { size: 11 }
        },
        hovermode: 'x unified'
      };

      const config = { responsive: true, displayModeBar: false };
      
      // Clean and render Plotly
      await Plotly.react(chartEl, [traceIF, traceLOF, traceConsensus], layout, config);

      // Hide loading state only after successful render
      if (loadingEl) loadingEl.classList.add('hidden');
      if (errorEl) errorEl.classList.add('hidden');
      if (emptyEl) emptyEl.classList.add('hidden');
    } catch (err) {
      console.error('Failed to load investigation timeline:', err);
      if (loadingEl) loadingEl.classList.add('hidden');
      if (errorEl) {
        errorEl.classList.remove('hidden');
        const msgEl = document.getElementById('timeline-error-msg');
        if (msgEl) msgEl.textContent = err.message || 'Failed to load timeline data';
      }
    }
  }

  // ---------------------------------------------------------------------------
  // 2. Observation Inspector & Climatological Context
  // ---------------------------------------------------------------------------
  async function inspectObservation(station, date) {
    if (!station || !date) return;
    state.currentStation = station;
    state.currentDate = date;

    try {
      const url = `/api/investigation/observation/${encodeURIComponent(station)}/${encodeURIComponent(date)}`;
      const res = await fetch(url);
      const json = await res.json();
      if (!json.success || !json.data) return;

      const pkg = json.data;
      const obs = pkg.observation;
      const pct = pkg.percentiles;
      const ctx = pkg.context_window;

      // Update Inspector Identity Header
      const titleEl = document.getElementById('inspector-title');
      const subEl = document.getElementById('inspector-subtitle');
      const badgeEl = document.getElementById('inspector-badge-status');
      const sevEl = document.getElementById('inspector-severity-val');

      if (titleEl) titleEl.textContent = `${obs.station_id} // ${obs.date} (${obs.station_name})`;
      if (subEl) subEl.textContent = `${obs.season} ${obs.year} • Lat: ${obs.latitude}° • Lon: ${obs.longitude}° • Elev: ${obs.elevation_m}m`;
      if (sevEl) sevEl.textContent = obs.severity_index !== null ? obs.severity_index.toFixed(4) : 'N/A';

      if (badgeEl) {
        badgeEl.textContent = obs.investigation_status;
        badgeEl.className = 'px-2 py-0.5 rounded text-[10px] font-mono font-medium border ';
        if (obs.investigation_status === 'Possible Genuine Meteorological Extreme') {
          badgeEl.className += 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20';
        } else if (obs.investigation_status.includes('Consensus')) {
          badgeEl.className += 'bg-purple-500/10 text-purple-400 border-purple-500/20';
        } else if (obs.investigation_status.includes('Measurement')) {
          badgeEl.className += 'bg-rose-500/10 text-rose-400 border-rose-500/20';
        } else {
          badgeEl.className += 'bg-cyan-500/10 text-cyan-400 border-cyan-500/20';
        }
      }

      // Update Physical Readings
      setText('insp-tmax', `${obs.tmax} °C`);
      setText('insp-tmin', `${obs.tmin} °C`);
      setText('insp-range', `${obs.temp_range} °C`);
      setText('insp-prcp', `${obs.prcp} mm`);
      setText('insp-snow', `${obs.snow} mm`);
      setText('insp-snwd', `${obs.snwd} mm`);
      setText('insp-diff', `${obs.tmax_diff_1d} °C`);

      // Update Diagnostics
      setText('insp-if-score', obs.if_score.toFixed(4));
      setText('insp-lof-score', obs.lof_score.toFixed(4));
      setText('insp-con-label', obs.consensus_label);
      setText('insp-qc-status', obs.qc_flagged ? 'Flagged Violation' : 'Passed (Clean)');

      const ifPredEl = document.getElementById('insp-if-pred');
      if (ifPredEl) {
        ifPredEl.textContent = obs.if_anomaly === 1 ? 'Anomaly (-1)' : 'Normal (1)';
        ifPredEl.className = obs.if_anomaly === 1 ? 'px-1.5 py-0.5 rounded text-[10px] font-bold bg-cyan-500/10 text-cyan-400' : 'text-[var(--text-muted)]';
      }

      const lofPredEl = document.getElementById('insp-lof-pred');
      if (lofPredEl) {
        lofPredEl.textContent = obs.lof_anomaly === 1 ? 'Anomaly (-1)' : 'Normal (1)';
        lofPredEl.className = obs.lof_anomaly === 1 ? 'px-1.5 py-0.5 rounded text-[10px] font-bold bg-amber-500/10 text-amber-400' : 'text-[var(--text-muted)]';
      }

      // Update Percentiles
      if (pct) {
        updatePercentileRow('tmin', pct.TMIN);
        updatePercentileRow('tmax', pct.TMAX);
        updatePercentileRow('snwd', pct.SNWD);
        updatePercentileRow('prcp', pct.PRCP);
      }

      // Update Narrative
      setText('inspector-rationale-text', pkg.scientific_rationale);
      const actionEl = document.getElementById('inspector-action-text');
      if (actionEl) {
        if (obs.is_known_extreme) {
          actionEl.textContent = 'Retain in Climate Archive (Verified Atmospheric Extreme)';
          actionEl.className = 'text-xs font-semibold text-emerald-400';
        } else if (obs.qc_flagged) {
          actionEl.textContent = 'Flag for Instrument Maintenance (Suspect Quality Violations)';
          actionEl.className = 'text-xs font-semibold text-rose-400';
        } else if (obs.consensus_label === 'Consensus Anomaly') {
          actionEl.textContent = 'Prioritize for Meteorological Peer Review';
          actionEl.className = 'text-xs font-semibold text-purple-400';
        } else {
          actionEl.textContent = 'Standard Observation Monitoring';
          actionEl.className = 'text-xs font-semibold text-[var(--text-secondary)]';
        }
      }

      // Render Context Window (7 Days)
      renderContextWindow(ctx, date);

      // Scroll smoothly to inspector
      const card = document.getElementById('observation-inspector-card');
      if (card) {
        card.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }

    } catch (err) {
      console.error('Failed to load observation details:', err);
    }
  }

  function setText(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
  }

  function updatePercentileRow(key, data) {
    if (!data) return;
    const valEl = document.getElementById(`pct-${key}-val`);
    const barEl = document.getElementById(`pct-${key}-bar`);
    if (valEl) {
      valEl.textContent = `${data.percentile_rank}%ile (Value: ${data.value})`;
    }
    if (barEl) {
      const clampPct = Math.max(2, Math.min(100, data.percentile_rank));
      barEl.style.width = `${clampPct}%`;
      if (clampPct > 95 || clampPct < 5) {
        barEl.className = 'bg-rose-400 h-full rounded-full transition-all duration-500';
      } else {
        barEl.className = 'bg-cyan-400 h-full rounded-full transition-all duration-500';
      }
    }
  }

  function renderContextWindow(days, targetDate) {
    const grid = document.getElementById('inspector-context-window-grid');
    if (!grid || !Array.isArray(days)) return;

    grid.innerHTML = days.map(d => {
      const isTarget = (d.date === targetDate);
      const isAnom = (d.if_anomaly === 1 || d.lof_anomaly === 1);
      const cardBorder = isTarget ? 'border-cyan-400 ring-2 ring-cyan-400/30 bg-[var(--bg-card-hover)]' : (isAnom ? 'border-purple-500/40 bg-[var(--bg-app)]' : 'border-[var(--border-subtle)] bg-[var(--bg-app)]');

      return `
        <div class="p-2.5 rounded-lg border ${cardBorder} text-xs font-mono space-y-1 transition-all">
          <div class="flex items-center justify-between text-[10px] ${isTarget ? 'text-cyan-400 font-bold' : 'text-[var(--text-muted)]'}">
            <span>${d.date.slice(5)}</span>
            ${isTarget ? '<span class="px-1 rounded bg-cyan-400/20 text-cyan-400 font-bold">TARGET</span>' : (isAnom ? '<span class="text-purple-400 font-bold">ANOM</span>' : '')}
          </div>
          <div class="flex justify-between text-[11px] font-semibold">
            <span class="text-[var(--text-primary)]">${d.tmax}° / <span class="text-cyan-400">${d.tmin}°</span></span>
          </div>
          <div class="flex justify-between text-[10px] text-[var(--text-secondary)]">
            <span>Prcp: ${d.prcp}mm</span>
            ${d.snow > 0 ? `<span class="text-cyan-300">Snw: ${d.snow}</span>` : ''}
          </div>
        </div>
      `;
    }).join('');
  }

  // ---------------------------------------------------------------------------
  // 3. Observation Ledger & Filtering
  // ---------------------------------------------------------------------------
  async function loadObservations() {
    const tbody = document.getElementById('ledger-table-body');
    const badgeEl = document.getElementById('ledger-count-badge');
    if (!tbody) return;

    tbody.innerHTML = `
      <tr>
        <td colspan="12" class="py-8 text-center text-xs font-mono text-[var(--text-muted)]">
          <div class="flex items-center justify-center gap-2">
            <i data-lucide="loader" class="w-4 h-4 animate-spin text-cyan-400"></i>
            <span>Loading observation records...</span>
          </div>
        </td>
      </tr>
    `;
    if (window.lucide) lucide.createIcons();

    try {
      const url = new URL('/api/investigation/observations', window.location.origin);
      url.searchParams.set('page', state.page);
      url.searchParams.set('per_page', state.perPage);
      url.searchParams.set('sort_by', state.sortBy);
      url.searchParams.set('sort_dir', state.sortDir);

      if (state.station !== 'ALL') url.searchParams.set('station', state.station);
      if (state.classification !== 'all') url.searchParams.set('classification', state.classification);
      if (state.year !== 'ALL') url.searchParams.set('year', state.year);
      if (state.season !== 'ALL') url.searchParams.set('season', state.season);
      if (state.qflag !== 'all') url.searchParams.set('qflag', state.qflag);
      if (state.search) url.searchParams.set('search', state.search);

      const res = await fetch(url.toString());
      const json = await res.json();
      if (!json.success || !json.data) return;

      const d = json.data;
      state.totalPages = d.pages;

      if (badgeEl) {
        badgeEl.textContent = `Showing ${d.items.length} of ${d.total.toLocaleString()} records (Page ${d.page} of ${d.pages})`;
      }

      if (d.items.length === 0) {
        tbody.innerHTML = `
          <tr>
            <td colspan="12" class="py-8 text-center text-xs font-mono text-[var(--text-muted)]">
              No observation records match the selected filters.
            </td>
          </tr>
        `;
        renderPagination(d);
        return;
      }

      tbody.innerHTML = d.items.map(row => {
        let statusBadgeClass = 'bg-cyan-500/10 text-cyan-400 border-cyan-500/20';
        if (row.investigation_status === 'Possible Genuine Meteorological Extreme') {
          statusBadgeClass = 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20';
        } else if (row.investigation_status.includes('Consensus')) {
          statusBadgeClass = 'bg-purple-500/10 text-purple-400 border-purple-500/20';
        } else if (row.investigation_status.includes('Measurement')) {
          statusBadgeClass = 'bg-rose-500/10 text-rose-400 border-rose-500/20';
        }

        return `
          <tr class="hover:bg-[var(--bg-card-hover)] transition-colors cursor-pointer" onclick="AtmosphereInvestigation.inspectObservation('${row.station_id}', '${row.date}')">
            <td class="py-2.5 px-3 font-mono">
              <span class="font-bold text-[var(--text-primary)]">${row.date}</span>
              <span class="block text-[10px] text-[var(--text-muted)]">${row.station_id} • ${row.station_name.split(',')[0]}</span>
            </td>
            <td class="py-2.5 px-3 text-[11px] text-[var(--text-secondary)]">${row.season}</td>
            <td class="py-2.5 px-3 text-right font-mono">${row.tmax}</td>
            <td class="py-2.5 px-3 text-right font-mono text-cyan-400">${row.tmin}</td>
            <td class="py-2.5 px-3 text-right font-mono ${row.prcp > 25 ? 'font-bold text-blue-400' : ''}">${row.prcp}</td>
            <td class="py-2.5 px-3 text-right font-mono ${row.snow > 0 ? 'font-bold text-cyan-300' : ''}">${row.snow}</td>
            <td class="py-2.5 px-3 text-center font-mono">
              <span class="px-1.5 py-0.5 rounded text-[10px] font-bold ${row.if_anomaly === 1 ? 'bg-cyan-500/20 text-cyan-400' : 'text-[var(--text-muted)]'}">
                ${row.if_anomaly === 1 ? '-1' : '1'}
              </span>
            </td>
            <td class="py-2.5 px-3 text-center font-mono">
              <span class="px-1.5 py-0.5 rounded text-[10px] font-bold ${row.lof_anomaly === 1 ? 'bg-amber-500/20 text-amber-400' : 'text-[var(--text-muted)]'}">
                ${row.lof_anomaly === 1 ? '-1' : '1'}
              </span>
            </td>
            <td class="py-2.5 px-3 text-center font-mono">
              <span class="px-1.5 py-0.5 rounded text-[10px] font-bold ${row.consensus_label === 'Consensus Anomaly' ? 'bg-purple-500/20 text-purple-400' : 'text-[var(--text-muted)]'}">
                ${row.consensus_label === 'Consensus Anomaly' ? 'Both' : (row.consensus_label === 'Normal' ? 'Normal' : 'Single')}
              </span>
            </td>
            <td class="py-2.5 px-3 text-right font-mono font-bold text-purple-400">
              ${row.severity_index !== null ? row.severity_index.toFixed(4) : '0.0000'}
            </td>
            <td class="py-2.5 px-3">
              <span class="px-2 py-0.5 rounded text-[10px] font-mono border ${statusBadgeClass}">
                ${row.investigation_status}
              </span>
            </td>
            <td class="py-2.5 px-3 text-right">
              <button
                class="px-2 py-1 rounded bg-[var(--bg-app)] hover:bg-[var(--bg-surface)] text-cyan-400 border border-[var(--border-subtle)] font-mono text-[10px] transition-colors"
                onclick="event.stopPropagation(); AtmosphereInvestigation.inspectObservation('${row.station_id}', '${row.date}')"
              >
                Inspect
              </button>
            </td>
          </tr>
        `;
      }).join('');

      renderPagination(d);
      if (window.lucide) lucide.createIcons();

    } catch (err) {
      console.error('Failed to load observations ledger:', err);
    }
  }

  function renderPagination(d) {
    const info = document.getElementById('pagination-info');
    const btnPrev = document.getElementById('btn-page-prev');
    const btnNext = document.getElementById('btn-page-next');
    const pills = document.getElementById('pagination-pages-pills');

    if (info) info.textContent = `Page ${d.page} of ${d.pages} (${d.total.toLocaleString()} total)`;
    if (btnPrev) btnPrev.disabled = !d.has_prev;
    if (btnNext) btnNext.disabled = !d.has_next;

    if (pills) {
      pills.innerHTML = '';
      const startP = Math.max(1, d.page - 2);
      const endP = Math.min(d.pages, d.page + 2);
      for (let p = startP; p <= endP; p++) {
        const isCur = (p === d.page);
        const btn = document.createElement('button');
        btn.className = `w-7 h-7 rounded text-xs font-mono border transition-colors ${
          isCur ? 'bg-cyan-500 text-slate-950 font-bold border-cyan-400' : 'bg-[var(--bg-app)] text-[var(--text-secondary)] border-[var(--border-subtle)] hover:text-[var(--text-primary)]'
        }`;
        btn.textContent = p;
        btn.onclick = () => { state.page = p; loadObservations(); };
        pills.appendChild(btn);
      }
    }
  }

  function setSort(col) {
    if (state.sortBy === col) {
      state.sortDir = (state.sortDir === 'asc') ? 'desc' : 'asc';
    } else {
      state.sortBy = col;
      state.sortDir = 'desc';
    }
    state.page = 1;
    loadObservations();
  }

  // ---------------------------------------------------------------------------
  // 4. Bind Events & Lifecycle
  // ---------------------------------------------------------------------------
  function bindEvents() {
    // Timeline filters
    const stTl = document.getElementById('timeline-filter-station');
    if (stTl) stTl.addEventListener('change', (e) => {
      state.timelineStation = e.target.value;
      loadTimelineChart();
    });

    const yrTl = document.getElementById('timeline-filter-year');
    if (yrTl) yrTl.addEventListener('change', (e) => {
      state.timelineYear = e.target.value;
      loadTimelineChart();
    });

    // Ledger filters
    const fSt = document.getElementById('filter-station');
    if (fSt) fSt.addEventListener('change', (e) => { state.station = e.target.value; state.page = 1; loadObservations(); });

    const fCl = document.getElementById('filter-classification');
    if (fCl) fCl.addEventListener('change', (e) => { state.classification = e.target.value; state.page = 1; loadObservations(); });

    const fYr = document.getElementById('filter-year');
    if (fYr) fYr.addEventListener('change', (e) => { state.year = e.target.value; state.page = 1; loadObservations(); });

    const fSe = document.getElementById('filter-season');
    if (fSe) fSe.addEventListener('change', (e) => { state.season = e.target.value; state.page = 1; loadObservations(); });

    const fQc = document.getElementById('filter-qflag');
    if (fQc) fQc.addEventListener('change', (e) => { state.qflag = e.target.value; state.page = 1; loadObservations(); });

    let searchTimer = null;
    const fSr = document.getElementById('filter-search');
    if (fSr) fSr.addEventListener('input', (e) => {
      clearTimeout(searchTimer);
      searchTimer = setTimeout(() => {
        state.search = e.target.value.trim();
        state.page = 1;
        loadObservations();
      }, 350);
    });

    // Pagination buttons
    const btnPrev = document.getElementById('btn-page-prev');
    if (btnPrev) btnPrev.addEventListener('click', () => {
      if (state.page > 1) { state.page--; loadObservations(); }
    });

    const btnNext = document.getElementById('btn-page-next');
    if (btnNext) btnNext.addEventListener('click', () => {
      if (state.page < state.totalPages) { state.page++; loadObservations(); }
    });

    // Export CSV
    const btnExp = document.getElementById('btn-export-anomalies-csv');
    if (btnExp) btnExp.addEventListener('click', () => {
      const url = new URL('/api/investigation/export', window.location.origin);
      if (state.station !== 'ALL') url.searchParams.set('station', state.station);
      if (state.classification !== 'all') url.searchParams.set('classification', state.classification);
      if (state.year !== 'ALL') url.searchParams.set('year', state.year);
      if (state.season !== 'ALL') url.searchParams.set('season', state.season);
      if (state.qflag !== 'all') url.searchParams.set('qflag', state.qflag);
      if (state.search) url.searchParams.set('search', state.search);
      window.location.href = url.toString();
    });

    // Extreme events buttons
    document.querySelectorAll('.btn-inspect-extreme').forEach(btn => {
      btn.addEventListener('click', () => {
        const st = btn.getAttribute('data-station');
        const dt = btn.getAttribute('data-date');
        inspectObservation(st, dt);
      });
    });

    // Timeline retry button
    const btnTlRetry = document.getElementById('btn-timeline-retry');
    if (btnTlRetry) btnTlRetry.addEventListener('click', () => {
      loadTimelineChart();
    });

    // Window resize handler for responsive chart resizing
    let resizeTimer = null;
    window.addEventListener('resize', () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => {
        const el = document.getElementById('chart-investigation-timeline');
        if (el && typeof Plotly !== 'undefined' && el.data) {
          Plotly.Plots.resize(el);
        }
      }, 100);
    });

    // Theme changes
    const observer = new MutationObserver(() => {
      state.isDark = document.documentElement.getAttribute('data-theme') !== 'light';
      loadTimelineChart();
    });
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  }

  function init() {
    bindEvents();
    loadTimelineChart();
    loadObservations();
    // Default inspector to Chicago Polar Vortex Day 2 (2019-01-30)
    inspectObservation('USW00094846', '2019-01-30');
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  return {
    inspectObservation,
    setSort
  };
})();


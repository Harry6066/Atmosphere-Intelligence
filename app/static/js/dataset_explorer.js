/**
 * Atmosphere Intelligence — Dataset Explorer Controller
 * Part of INT396 Unsupervised Machine Learning Project (SIH26073).
 * Manages Leaflet.js weather station mapping, multi-criteria filtering,
 * server-side paginated observation rendering, and dynamic CSV export.
 */

const AtmosphereExplorer = (function () {
  'use strict';

  let map = null;
  let stationMarkers = {};
  let currentSortCol = 'date';
  let currentSortDir = 'desc';
  let currentPage = 1;
  let currentPageSize = 25;
  let totalRecords = 11688;
  let searchTimeout = null;

  // Initialize
  function init() {
    initMap();
    bindFilterEvents();

    // Check for query parameters passed from global search or cross-module links
    const urlParams = new URLSearchParams(window.location.search);
    const searchParam = urlParams.get('search');
    if (searchParam) {
      const searchInput = document.getElementById('search-filter');
      if (searchInput) searchInput.value = searchParam;
    }
    const stationParam = urlParams.get('station');
    if (stationParam) {
      const stationSelect = document.getElementById('station-filter');
      if (stationSelect) stationSelect.value = stationParam;
      highlightStationCard(stationParam);
    }

    loadRecords(1);

    window.addEventListener('resize', () => {
      if (map) map.invalidateSize();
    });
  }

  // ---------------------------------------------------------------------------
  // 1. Leaflet.js Interactive Weather Station Map
  // ---------------------------------------------------------------------------
  function initMap() {
    const mapContainer = document.getElementById('station-map');
    if (!mapContainer || typeof L === 'undefined') return;

    // Center on contiguous United States
    map = L.map('station-map', {
      center: [38.5, -96.0],
      zoom: 4,
      zoomControl: true,
      attributionControl: true
    });

    // Standard OpenStreetMap raster tile layer
    L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener noreferrer">OpenStreetMap</a> contributors'
    }).addTo(map);

    // Station Coordinates from NOAA Dataset
    const stations = window.ATMOSPHERE_STATIONS || [
      { station_id: 'USW00094728', name: 'NEW YORK CENTRAL PARK, NY', latitude: 40.77898, longitude: -73.96925, elevation_m: 42.7, records_count: 2922, consensus_anomalies: 16 },
      { station_id: 'USW00094846', name: 'CHICAGO OHARE INTL AP, IL', latitude: 41.96018, longitude: -87.93164, elevation_m: 204.8, records_count: 2922, consensus_anomalies: 58 },
      { station_id: 'USW00024233', name: 'SEATTLE TACOMA INTL AP, WA', latitude: 47.44467, longitude: -122.31442, elevation_m: 112.5, records_count: 2922, consensus_anomalies: 2 },
      { station_id: 'USW00012839', name: 'MIAMI INTERNATIONAL AP, FL', latitude: 25.78806, longitude: -80.31694, elevation_m: 2.1, records_count: 2922, consensus_anomalies: 2 }
    ];

    const bounds = [];

    stations.forEach(stn => {
      const lat = stn.latitude;
      const lon = stn.longitude;
      bounds.push([lat, lon]);

      // Custom Glowing Cyan DivIcon
      const pinIcon = L.divIcon({
        className: 'atm-map-pin',
        iconSize: [20, 20],
        iconAnchor: [10, 10],
        popupAnchor: [0, -12]
      });

      const marker = L.marker([lat, lon], { icon: pinIcon }).addTo(map);

      // Rich HTML Popup
      const popupHtml = `
        <div style="font-family: 'Inter', sans-serif; min-width: 200px; padding: 4px;">
          <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 4px;">
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 10px; font-weight: bold; color: #00d4ff; background: rgba(0,212,255,0.1); padding: 2px 6px; border-radius: 4px;">${stn.station_id}</span>
            <span style="font-size: 10px; color: #94a3b8; font-family: 'JetBrains Mono', monospace;">${stn.elevation_m}m elev</span>
          </div>
          <div style="font-weight: 700; font-size: 12px; margin-bottom: 4px;">${stn.name}</div>
          <div style="font-size: 11px; color: #94a3b8; margin-bottom: 8px;">
            Coords: ${lat.toFixed(4)}°, ${lon.toFixed(4)}°<br>
            Records: <strong>${stn.records_count}</strong> (2016–2023)<br>
            Consensus Outliers: <strong style="color: #00d4ff;">${stn.consensus_anomalies}</strong>
          </div>
          <button onclick="AtmosphereExplorer.selectStation('${stn.station_id}')"
                  style="width: 100%; font-size: 11px; background: #00d4ff; color: #070d18; font-weight: 600; padding: 5px 8px; border-radius: 4px; border: none; cursor: pointer;">
            Filter Table to Station
          </button>
        </div>
      `;

      marker.bindPopup(popupHtml);
      stationMarkers[stn.station_id] = marker;
    });

    if (bounds.length > 0) {
      map.fitBounds(bounds, { padding: [50, 50], maxZoom: 6 });
    }
  }

  function resetMapBounds() {
    if (!map) return;
    const stations = window.ATMOSPHERE_STATIONS || [];
    if (stations.length === 0) return;
    const bounds = stations.map(s => [s.latitude, s.longitude]);
    map.fitBounds(bounds, { padding: [50, 50], maxZoom: 6 });
  }

  // ---------------------------------------------------------------------------
  // 2. Multi-Criteria Filter Event Binding
  // ---------------------------------------------------------------------------
  function bindFilterEvents() {
    const stationSelect = document.getElementById('station-filter');
    const yearSelect = document.getElementById('year-filter');
    const startDateInput = document.getElementById('start-date-filter');
    const endDateInput = document.getElementById('end-date-filter');
    const anomalySelect = document.getElementById('anomaly-filter');
    const searchInput = document.getElementById('search-filter');
    const pageSizeSelect = document.getElementById('page-size-filter');
    const resetBtn = document.getElementById('btn-reset-filters');

    if (stationSelect) {
      stationSelect.addEventListener('change', () => {
        highlightStationCard(stationSelect.value);
        loadRecords(1);
      });
    }

    if (yearSelect) {
      yearSelect.addEventListener('change', () => {
        // If year is picked, clear start/end date inputs to avoid conflicts
        if (yearSelect.value !== 'ALL') {
          if (startDateInput) startDateInput.value = '';
          if (endDateInput) endDateInput.value = '';
        }
        loadRecords(1);
      });
    }

    if (startDateInput) {
      startDateInput.addEventListener('change', () => {
        if (yearSelect && startDateInput.value) yearSelect.value = 'ALL';
        loadRecords(1);
      });
    }

    if (endDateInput) {
      endDateInput.addEventListener('change', () => {
        if (yearSelect && endDateInput.value) yearSelect.value = 'ALL';
        loadRecords(1);
      });
    }

    if (anomalySelect) {
      anomalySelect.addEventListener('change', () => loadRecords(1));
    }

    if (pageSizeSelect) {
      pageSizeSelect.addEventListener('change', () => {
        currentPageSize = parseInt(pageSizeSelect.value, 10) || 25;
        loadRecords(1);
      });
    }

    if (searchInput) {
      searchInput.addEventListener('input', () => {
        clearTimeout(searchTimeout);
        searchTimeout = setTimeout(() => {
          loadRecords(1);
        }, 300);
      });
    }

    if (resetBtn) {
      resetBtn.addEventListener('click', resetFilters);
    }

    // Pagination buttons
    const btnFirst = document.getElementById('btn-page-first');
    const btnPrev = document.getElementById('btn-page-prev');
    const btnNext = document.getElementById('btn-page-next');
    const btnLast = document.getElementById('btn-page-last');

    if (btnFirst) btnFirst.addEventListener('click', () => loadRecords(1));
    if (btnPrev) btnPrev.addEventListener('click', () => loadRecords(currentPage - 1));
    if (btnNext) btnNext.addEventListener('click', () => loadRecords(currentPage + 1));
    if (btnLast) {
      btnLast.addEventListener('click', () => {
        const totalPages = Math.max(1, Math.ceil(totalRecords / currentPageSize));
        loadRecords(totalPages);
      });
    }
  }

  // ---------------------------------------------------------------------------
  // 3. Selection & Highlight Helpers
  // ---------------------------------------------------------------------------
  function selectStation(stationId) {
    const stationSelect = document.getElementById('station-filter');
    if (stationSelect) {
      stationSelect.value = stationId;
      highlightStationCard(stationId);
      loadRecords(1);

      // Pan map to station
      if (map && stationMarkers[stationId]) {
        const marker = stationMarkers[stationId];
        map.setView(marker.getLatLng(), 8, { animate: true });
        marker.openPopup();
      }

      if (window.Atmosphere) {
        window.Atmosphere.toast(`Filtered to station: ${stationId}`, 'info');
      }
    }
  }

  function highlightStationCard(stationId) {
    document.querySelectorAll('.station-card').forEach(card => {
      if (stationId !== 'ALL' && card.getAttribute('data-station') === stationId) {
        card.classList.add('border-cyan-400', 'shadow-[0_0_15px_rgba(0,212,255,0.15)]');
      } else {
        card.classList.remove('border-cyan-400', 'shadow-[0_0_15px_rgba(0,212,255,0.15)]');
      }
    });
  }

  function resetFilters() {
    const stationSelect = document.getElementById('station-filter');
    const yearSelect = document.getElementById('year-filter');
    const startDateInput = document.getElementById('start-date-filter');
    const endDateInput = document.getElementById('end-date-filter');
    const anomalySelect = document.getElementById('anomaly-filter');
    const searchInput = document.getElementById('search-filter');
    const pageSizeSelect = document.getElementById('page-size-filter');

    if (stationSelect) stationSelect.value = 'ALL';
    if (yearSelect) yearSelect.value = 'ALL';
    if (startDateInput) startDateInput.value = '';
    if (endDateInput) endDateInput.value = '';
    if (anomalySelect) anomalySelect.value = 'ALL';
    if (searchInput) searchInput.value = '';
    if (pageSizeSelect) pageSizeSelect.value = '25';

    highlightStationCard('ALL');
    resetMapBounds();
    loadRecords(1);

    if (window.Atmosphere) {
      window.Atmosphere.toast('Filters reset to all records', 'normal');
    }
  }

  function setSort(colName) {
    if (currentSortCol === colName) {
      currentSortDir = currentSortDir === 'asc' ? 'desc' : 'asc';
    } else {
      currentSortCol = colName;
      currentSortDir = 'desc';
    }
    loadRecords(currentPage);
  }

  // ---------------------------------------------------------------------------
  // 4. Data Fetching & Table Rendering
  // ---------------------------------------------------------------------------
  function loadRecords(page) {
    const tableLoading = document.getElementById('table-loading');
    const tableEmpty = document.getElementById('table-empty');
    const tableBody = document.getElementById('table-body');
    const exportBtn = document.getElementById('btn-export-csv');

    if (tableLoading) tableLoading.classList.remove('hidden');
    if (tableEmpty) tableEmpty.classList.add('hidden');

    const station = document.getElementById('station-filter')?.value || 'ALL';
    const year = document.getElementById('year-filter')?.value || 'ALL';
    const startDate = document.getElementById('start-date-filter')?.value || '';
    const endDate = document.getElementById('end-date-filter')?.value || '';
    const anomaly = document.getElementById('anomaly-filter')?.value || 'ALL';
    const search = document.getElementById('search-filter')?.value || '';

    const params = new URLSearchParams();
    params.set('page', page);
    params.set('page_size', currentPageSize);
    params.set('sort_col', currentSortCol);
    params.set('sort_dir', currentSortDir);

    if (station !== 'ALL') params.set('station', station);
    if (year !== 'ALL') params.set('year', year);
    if (startDate) params.set('start_date', startDate);
    if (endDate) params.set('end_date', endDate);
    if (anomaly !== 'ALL') params.set('anomaly_type', anomaly);
    if (search.trim()) params.set('search', search.trim());

    // Update CSV export link
    if (exportBtn) {
      const exportParams = new URLSearchParams(params);
      exportParams.delete('page');
      exportParams.delete('page_size');
      exportBtn.href = `/api/dataset/export?${exportParams.toString()}`;
    }

    fetch(`/api/dataset/records?${params.toString()}`)
      .then(res => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then(json => {
        if (tableLoading) tableLoading.classList.add('hidden');
        if (!json.success) throw new Error(json.error || 'API error');

        const records = json.data || [];
        const pagination = json.pagination || {};
        const summary = json.summary || {};

        currentPage = pagination.page || 1;
        totalRecords = pagination.total_records || 0;

        // Update Summary Counters
        updateSummaryCounters(summary);

        // Update Table Rows
        renderTableRows(records);

        // Update Pagination Controls
        updatePaginationControls(pagination);

        // Re-initialize Lucide icons in table
        if (window.lucide) {
          window.lucide.createIcons();
        }
      })
      .catch(err => {
        if (tableLoading) tableLoading.classList.add('hidden');
        console.error('Error fetching observations:', err);
        if (tableBody) {
          tableBody.innerHTML = `
            <tr>
              <td colspan="10" class="py-8 text-center text-rose-400 font-mono">
                Failed to load observation records: ${err.message}
              </td>
            </tr>
          `;
        }
      });
  }

  function updateSummaryCounters(summary) {
    const elFiltered = document.getElementById('stat-filtered-count');
    const elMeanTmax = document.getElementById('stat-mean-tmax');
    const elMeanTmin = document.getElementById('stat-mean-tmin');
    const elTotalPrcp = document.getElementById('stat-total-prcp');
    const elConsensus = document.getElementById('stat-consensus-count');

    if (elFiltered) elFiltered.textContent = (summary.filtered_count ?? totalRecords).toLocaleString();
    if (elMeanTmax) elMeanTmax.textContent = `${summary.mean_tmax ?? 0}°C`;
    if (elMeanTmin) elMeanTmin.textContent = `${summary.mean_tmin ?? 0}°C`;
    if (elTotalPrcp) elTotalPrcp.textContent = `${(summary.total_prcp ?? 0).toLocaleString()} mm`;
    if (elConsensus) elConsensus.textContent = summary.consensus_count ?? 0;
  }

  function renderTableRows(records) {
    const tableBody = document.getElementById('table-body');
    const tableEmpty = document.getElementById('table-empty');
    if (!tableBody) return;

    if (records.length === 0) {
      tableBody.innerHTML = '';
      if (tableEmpty) tableEmpty.classList.remove('hidden');
      return;
    }

    if (tableEmpty) tableEmpty.classList.add('hidden');

    const rowsHtml = records.map(r => {
      // Detection badge styling
      let statusBadge = '<span class="atm-badge text-[10px] bg-slate-800 text-slate-300">Normal</span>';
      if (r.consensus_label === 'Consensus Anomaly') {
        statusBadge = '<span class="atm-badge atm-badge-cyan text-[10px] font-bold">Consensus Outlier</span>';
      } else if (r.consensus_label === 'IF Only') {
        statusBadge = '<span class="atm-badge text-[10px] bg-sky-900/40 text-sky-300 border border-sky-500/40">IF Only</span>';
      } else if (r.consensus_label === 'LOF Only') {
        statusBadge = '<span class="atm-badge text-[10px] bg-indigo-900/40 text-indigo-300 border border-indigo-500/40">LOF Only</span>';
      }

      // Quality Flag Indicator
      let flagIndicator = '<span class="text-[var(--text-muted)] font-mono text-[11px]">—</span>';
      if (r.has_qc_flags) {
        flagIndicator = '<span class="text-amber-400 font-mono text-[11px]" title="QC Flag Present">Flagged</span>';
      }

      // Station Short Code
      const stnShort = r.station.replace('USW000', '');

      return `
        <tr class="hover:bg-[var(--bg-card-hover)] transition-colors">
          <td class="font-mono text-[var(--text-primary)] font-medium whitespace-nowrap">${r.date}</td>
          <td>
            <span class="atm-badge atm-badge-cyan font-mono text-[10px]" title="${r.name}">${r.station}</span>
          </td>
          <td class="font-mono ${r.tmax > 35 ? 'text-amber-400 font-bold' : (r.tmax < -10 ? 'text-cyan-400 font-bold' : '')}">${r.tmax.toFixed(1)}</td>
          <td class="font-mono ${r.tmin < -15 ? 'text-cyan-400 font-bold' : ''}">${r.tmin.toFixed(1)}</td>
          <td class="font-mono">${r.temp_range.toFixed(1)}</td>
          <td class="font-mono ${r.prcp > 50 ? 'text-sky-300 font-bold' : ''}">${r.prcp.toFixed(1)}</td>
          <td class="font-mono text-[var(--text-muted)]">${r.snow > 0 ? `<strong class="text-white">${r.snow.toFixed(1)}</strong>` : '0.0'}</td>
          <td class="font-mono text-[var(--text-muted)]">${r.snwd > 0 ? `<strong class="text-white">${r.snwd.toFixed(1)}</strong>` : '0.0'}</td>
          <td>${statusBadge}</td>
          <td>${flagIndicator}</td>
        </tr>
      `;
    }).join('');

    tableBody.innerHTML = rowsHtml;
  }

  function updatePaginationControls(pagination) {
    const page = pagination.page || 1;
    const totalPages = pagination.total_pages || 1;
    const totalRecordsCount = pagination.total_records || 0;
    const pageSize = pagination.page_size || 25;

    const startIdx = totalRecordsCount === 0 ? 0 : (page - 1) * pageSize + 1;
    const endIdx = Math.min(page * pageSize, totalRecordsCount);

    const infoEl = document.getElementById('pagination-info');
    const summaryEl = document.getElementById('pagination-summary');

    if (infoEl) infoEl.textContent = `Showing ${startIdx.toLocaleString()} to ${endIdx.toLocaleString()} of ${totalRecordsCount.toLocaleString()} records`;
    if (summaryEl) summaryEl.textContent = `Page ${page} of ${totalPages.toLocaleString()}`;

    const btnFirst = document.getElementById('btn-page-first');
    const btnPrev = document.getElementById('btn-page-prev');
    const btnNext = document.getElementById('btn-page-next');
    const btnLast = document.getElementById('btn-page-last');

    if (btnFirst) btnFirst.disabled = page <= 1;
    if (btnPrev) btnPrev.disabled = page <= 1;
    if (btnNext) btnNext.disabled = page >= totalPages;
    if (btnLast) btnLast.disabled = page >= totalPages;

    // Render Page Number Buttons (window around current page)
    const numbersContainer = document.getElementById('page-numbers-container');
    if (numbersContainer) {
      let pageBtnsHtml = '';
      const startPage = Math.max(1, page - 2);
      const endPage = Math.min(totalPages, page + 2);

      for (let p = startPage; p <= endPage; p++) {
        const isActive = p === page;
        pageBtnsHtml += `
          <button onclick="AtmosphereExplorer.loadRecords(${p})" 
                  class="px-2.5 py-1 rounded text-xs font-mono ${isActive ? 'bg-cyan-500 text-slate-950 font-bold' : 'bg-[var(--bg-surface)] text-[var(--text-secondary)] hover:text-[var(--text-primary)]'}">
            ${p}
          </button>
        `;
      }
      numbersContainer.innerHTML = pageBtnsHtml;
    }
  }

  // Public API
  return {
    init: init,
    selectStation: selectStation,
    resetFilters: resetFilters,
    resetMapBounds: resetMapBounds,
    setSort: setSort,
    loadRecords: loadRecords
  };

})();

// Initialize on DOMContentLoaded
document.addEventListener('DOMContentLoaded', AtmosphereExplorer.init);


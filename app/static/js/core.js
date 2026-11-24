/**
 * ATMOSPHERE INTELLIGENCE — CORE PLATFORM JAVASCRIPT
 * Course: INT396 (Unsupervised ML) | Problem Statement: SIH26073
 * Handles theme switching, sidebar collapse, mobile drawer, toasts, and hotkeys.
 */

(function () {
  'use strict';

  // Global namespace
  window.Atmosphere = window.Atmosphere || {};

  // --- 1. THEME SWITCHER SYSTEM ---
  const THEME_STORAGE_KEY = 'atmosphere-theme';

  function initTheme() {
    const savedTheme = localStorage.getItem(THEME_STORAGE_KEY);
    const systemPrefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    const initialTheme = savedTheme || (systemPrefersDark ? 'dark' : 'dark'); // Default to dark-first

    applyTheme(initialTheme);

    const themeToggleBtn = document.getElementById('theme-toggle-btn');
    if (themeToggleBtn) {
      themeToggleBtn.addEventListener('click', toggleTheme);
    }
  }

  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem(THEME_STORAGE_KEY, theme);

    // Update Lucide icon on toggle button if present
    const iconContainer = document.getElementById('theme-toggle-icon');
    if (iconContainer) {
      iconContainer.innerHTML = theme === 'dark' 
        ? '<i data-lucide="sun" class="w-5 h-5 text-amber-400"></i>'
        : '<i data-lucide="moon" class="w-5 h-5 text-slate-700"></i>';
      if (window.lucide) {
        window.lucide.createIcons();
      }
    }
  }

  function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    applyTheme(newTheme);
    Atmosphere.toast(`Switched to ${newTheme === 'dark' ? 'Midnight Dark' : 'Daylight'} theme`, 'info', 2000);
  }

  // --- 2. SIDEBAR COLLAPSE & MOBILE DRAWER ---
  const SIDEBAR_STORAGE_KEY = 'atmosphere-sidebar-collapsed';

  function initSidebar() {
    // Restore desktop collapsed state
    const isCollapsed = localStorage.getItem(SIDEBAR_STORAGE_KEY) === 'true';
    if (isCollapsed && window.innerWidth >= 1024) {
      document.body.classList.add('sidebar-collapsed');
    }

    const desktopToggleBtn = document.getElementById('sidebar-toggle-btn');
    if (desktopToggleBtn) {
      desktopToggleBtn.addEventListener('click', () => {
        const collapsed = document.body.classList.toggle('sidebar-collapsed');
        localStorage.setItem(SIDEBAR_STORAGE_KEY, collapsed);
        if (window.lucide) window.lucide.createIcons();
      });
    }

    // Mobile Drawer
    const mobileMenuBtn = document.getElementById('mobile-menu-btn');
    const mobileBackdrop = document.getElementById('mobile-backdrop');
    const mobileSidebar = document.getElementById('mobile-sidebar');
    const closeMobileBtn = document.getElementById('mobile-sidebar-close');

    function openMobileDrawer() {
      if (mobileSidebar && mobileBackdrop) {
        mobileBackdrop.classList.remove('hidden');
        mobileSidebar.classList.remove('-translate-x-full');
        document.body.style.overflow = 'hidden';
      }
    }

    function closeMobileDrawer() {
      if (mobileSidebar && mobileBackdrop) {
        mobileSidebar.classList.add('-translate-x-full');
        mobileBackdrop.classList.add('hidden');
        document.body.style.overflow = '';
      }
    }

    if (mobileMenuBtn) mobileMenuBtn.addEventListener('click', openMobileDrawer);
    if (closeMobileBtn) closeMobileBtn.addEventListener('click', closeMobileDrawer);
    if (mobileBackdrop) mobileBackdrop.addEventListener('click', closeMobileDrawer);

    if (mobileSidebar) {
      const mobileNavLinks = mobileSidebar.querySelectorAll('.atm-nav-link');
      mobileNavLinks.forEach(link => {
        link.addEventListener('click', closeMobileDrawer);
      });
    }

    window.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') closeMobileDrawer();
    });
  }

  // --- 3. TOAST NOTIFICATION ENGINE ---
  Atmosphere.toast = function (message, type = 'info', duration = 3500) {
    let container = document.getElementById('atm-toast-container');
    if (!container) {
      container = document.createElement('div');
      container.id = 'atm-toast-container';
      document.body.appendChild(container);
    }

    const toast = document.createElement('div');
    toast.className = 'atm-toast';

    let iconHtml = '<i data-lucide="info" class="w-5 h-5 text-cyan-400"></i>';
    if (type === 'success') {
      iconHtml = '<i data-lucide="check-circle-2" class="w-5 h-5 text-emerald-400"></i>';
    } else if (type === 'error') {
      iconHtml = '<i data-lucide="alert-triangle" class="w-5 h-5 text-rose-400"></i>';
    } else if (type === 'warning') {
      iconHtml = '<i data-lucide="alert-circle" class="w-5 h-5 text-amber-400"></i>';
    }

    toast.innerHTML = `
      <div class="flex-shrink-0">${iconHtml}</div>
      <div class="flex-1 text-sm font-medium">${message}</div>
      <button class="text-slate-400 hover:text-slate-200 ml-2" onclick="this.parentElement.remove()">
        <i data-lucide="x" class="w-4 h-4"></i>
      </button>
    `;

    container.appendChild(toast);
    if (window.lucide) window.lucide.createIcons();

    setTimeout(() => {
      toast.style.transition = 'opacity 200ms ease-out, transform 200ms ease-out';
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(8px)';
      setTimeout(() => toast.remove(), 250);
    }, duration);
  };

  // --- 4. GLOBAL SEARCH HOTKEY (CTRL+K) ---
  function initHotkeys() {
    window.addEventListener('keydown', (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        const searchInput = document.getElementById('global-search-input');
        if (searchInput) {
          searchInput.focus();
          searchInput.select();
        }
      }
    });

    const searchInput = document.getElementById('global-search-input');
    if (searchInput) {
      searchInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
          e.preventDefault();
          const query = searchInput.value.trim();
          if (query) {
            window.location.href = '/dataset-explorer?search=' + encodeURIComponent(query);
          }
        }
      });
    }
  }

  // --- 5. RESPONSIVE PLOTLY AUTORESIZE ---
  function initPlotlyAutoResize() {
    let resizeTimer = null;
    window.addEventListener('resize', () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => {
        if (typeof Plotly !== 'undefined') {
          document.querySelectorAll('.js-plotly-plot').forEach((chartEl) => {
            Plotly.Plots.resize(chartEl);
          });
        }
      }, 150);
    });
  }

  // --- INITIALIZE ON DOM READY ---
  document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    initSidebar();
    initHotkeys();
    initPlotlyAutoResize();
    if (window.lucide) {
      window.lucide.createIcons();
    }
  });

})();


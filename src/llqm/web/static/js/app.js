/* ===== Theme Toggle ===== */

(function() {
  const stored = localStorage.getItem('llqm-theme');
  const theme = stored || 'dark';
  document.documentElement.setAttribute('data-theme', theme);
  updateIcons(theme);
})();

function updateIcons(theme) {
  const darkIcon = document.getElementById('theme-icon-dark');
  const lightIcon = document.getElementById('theme-icon-light');
  if (darkIcon && lightIcon) {
    darkIcon.classList.toggle('hidden', theme !== 'dark');
    lightIcon.classList.toggle('hidden', theme !== 'light');
  }
}

function toggleTheme() {
  const current = document.documentElement.getAttribute('data-theme');
  const next = current === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem('llqm-theme', next);
  updateIcons(next);
}

/* Re-apply icons after htmx swaps (SSE 'done' replaces page content) */
document.addEventListener('htmx:afterSettle', function() {
  const theme = document.documentElement.getAttribute('data-theme') || 'dark';
  updateIcons(theme);
});

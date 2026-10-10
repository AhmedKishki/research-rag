"use strict";

const toggle = document.getElementById('lan-toggle');
if (toggle) toggle.addEventListener('click', async () => {
  toggle.disabled = true;
  const result = document.getElementById('lan-result');
  result.textContent = 'Changing LAN access…';
  try {
    const response = await fetch('/control/lan', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({enabled: toggle.dataset.enabled === 'true'})
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || 'LAN change failed');
    location.reload();
  } catch (error) {
    result.textContent = error.message;
    toggle.disabled = false;
  }
});

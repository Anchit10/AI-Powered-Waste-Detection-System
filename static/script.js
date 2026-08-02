document.addEventListener("DOMContentLoaded", () => {
    // ═══════ STATE ═══════
    let currentUser = null;
    let map = null;
    let markersLayer = null;
    let pickerMap = null;
    let pickerMarker = null;
    let reports = [];
    let localities = [];
    let authorities = {};
    let representatives = {};
    let currentFile = null;
    let lastAnalysis = null;
    let wasteChart = null;
    let areaChart = null;
    let isLocalitiesLoaded = false;

    const DEFAULT_LAT = 30.3165;
    const DEFAULT_LNG = 78.0322;
    const STAGES = ["Reported", "Taken Up", "Being Solved", "Solved"];

    // ═══════ DOM ═══════
    const sidebar = document.getElementById('sidebar');
    const sidebarOverlay = document.getElementById('sidebar-overlay');
    const hamburger = document.getElementById('hamburger');
    const navItems = document.querySelectorAll('.nav-item');
    const views = document.querySelectorAll('.view');
    const headerStat = document.getElementById('header-stat');
    const headerStatMobile = document.getElementById('header-stat-mobile');

    const fileInput = document.getElementById('file-input');
    const uploadZone = document.getElementById('upload-zone');
    const previewContainer = document.getElementById('preview-container');
    const previewImg = document.getElementById('preview-img');
    const btnClearPreview = document.getElementById('btn-clear-preview');
    const btnAnalyze = document.getElementById('btn-analyze');
    const reportLoader = document.getElementById('report-loader');
    const resultType = document.getElementById('result-type');
    const resultSeverityBadge = document.getElementById('result-severity-badge');
    const resultConfidence = document.getElementById('result-confidence');
    const resultImage = document.getElementById('result-image');
    const detectionSvg = document.getElementById('detection-svg');
    const detectionsList = document.getElementById('detections-list');
    const inputLat = document.getElementById('input-lat');
    const inputLng = document.getElementById('input-lng');
    const inputLocality = document.getElementById('input-locality');
    const btnGps = document.getElementById('btn-gps-locate');
    const btnSubmitReport = document.getElementById('btn-submit-report');
    const successMeta = document.getElementById('success-meta');
    const btnNewReport = document.getElementById('btn-new-report');
    const btnGoMap = document.getElementById('btn-go-map');
    const authorityInfoDiv = document.getElementById('responsible-authority-info');
    const authorityNameDisplay = document.getElementById('authority-name-display');
    const representativeNameDisplay = document.getElementById('representative-name-display');

    const authModal = document.getElementById('auth-modal');
    const btnAuthToggle = document.getElementById('btn-auth-toggle');
    const modalClose = document.getElementById('modal-close');
    const userMenu = document.getElementById('user-menu');
    const userEmailDisplay = document.getElementById('user-email-display');
    const userRoleBadge = document.getElementById('user-role-badge');
    const btnLogout = document.getElementById('btn-logout');
    const navAdmin = document.getElementById('nav-admin');

    const filterLocality = document.getElementById('filter-locality');
    const filterStatus = document.getElementById('filter-status');

    const reportModal = document.getElementById('report-modal');
    const reportModalClose = document.getElementById('report-modal-close');
    const reportModalBackdrop = document.getElementById('report-modal-backdrop');
    const reportDetailContent = document.getElementById('report-detail-content');

    const galleryAreaSelect = document.getElementById('gallery-area-select');
    const galleryGrid = document.getElementById('gallery-grid');

    // ═══════ INIT ═══════
    initSidebar();
    initMap();
    initAuth();
    initReportFlow();
    loadLocalities();
    loadReports();

    // ═══════ SIDEBAR NAV ═══════
    function initSidebar() {
        hamburger.addEventListener('click', () => {
            sidebar.classList.toggle('open');
            sidebarOverlay.classList.toggle('hidden');
        });
        sidebarOverlay.addEventListener('click', () => {
            sidebar.classList.remove('open');
            sidebarOverlay.classList.add('hidden');
        });
        navItems.forEach(item => {
            item.addEventListener('click', () => {
                const viewId = item.dataset.view;
                navItems.forEach(n => n.classList.remove('active'));
                item.classList.add('active');
                views.forEach(v => v.classList.toggle('active', v.id === viewId));
                sidebar.classList.remove('open');
                sidebarOverlay.classList.add('hidden');
                if (viewId === 'view-map' && map) setTimeout(() => map.invalidateSize(), 150);
                if (viewId === 'view-stats') loadStats();
                if (viewId === 'view-admin') loadAdminData();
                if (viewId === 'view-gallery') initGallery();
            });
        });
    }

    function switchToView(viewId) {
        navItems.forEach(n => n.classList.toggle('active', n.dataset.view === viewId));
        views.forEach(v => v.classList.toggle('active', v.id === viewId));
        if (viewId === 'view-map' && map) setTimeout(() => map.invalidateSize(), 150);
    }

    // ═══════ MAP ═══════
    function initMap() {
        map = L.map('map').setView([DEFAULT_LAT, DEFAULT_LNG], 13);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            maxZoom: 19, attribution: '© OpenStreetMap'
        }).addTo(map);
        markersLayer = L.layerGroup().addTo(map);
        filterLocality.addEventListener('change', renderMarkers);
        filterStatus.addEventListener('change', renderMarkers);
    }

    function renderMarkers() {
        if (!markersLayer) return;
        markersLayer.clearLayers();
        const locF = filterLocality.value, statusF = filterStatus.value;

        reports.forEach(r => {
            if (locF && r.locality !== locF) return;
            if (statusF && r.status !== statusF) return;

            const sc = r.status.toLowerCase().replace(/ /g, '-');
            const icon = L.divIcon({
                className: 'x',
                html: `<div class="custom-marker ${sc}"></div>`,
                iconSize: [16, 16], iconAnchor: [8, 8]
            });

            const popup = `
                <div class="popup-title">${r.waste_type || 'Unknown'}</div>
                <div class="popup-meta">${r.locality || '—'} · ${r.reported_at ? new Date(r.reported_at).toLocaleDateString() : ''}</div>
                <span class="status-badge ${sc}" style="margin-top:4px;display:inline-block;">${r.status}</span>
                <div class="popup-meta" style="margin-top:3px;">🏛️ ${r.authority_name || 'Unassigned'}</div>
                <div class="popup-meta" style="font-style:italic;">👤 ${r.representative_name || 'Area Representative'}</div>
                ${r.image_path ? `<img src="${r.image_path}" style="width:100%;margin-top:6px;border-radius:6px;max-height:90px;object-fit:cover;">` : ''}
                <button class="popup-btn" onclick="viewReportDetail(${r.id})">View Details</button>
            `;
            L.marker([r.lat, r.lng], { icon }).bindPopup(popup).addTo(markersLayer);
        });
    }

    // ═══════ DATA LOADING ═══════
    async function loadLocalities() {
        try {
            const res = await fetch('/api/localities');
            const data = await res.json();
            localities = data.localities;
            authorities = data.authorities;
            representatives = data.representatives;
            isLocalitiesLoaded = true;

            localities.forEach(loc => {
                [filterLocality, inputLocality, galleryAreaSelect].forEach(sel => {
                    if (!sel) return;
                    const opt = document.createElement('option');
                    opt.value = loc; opt.textContent = loc;
                    sel.appendChild(opt);
                });
            });

            const authLocSel = document.getElementById('auth-locality');
            if (authLocSel) {
                localities.forEach(loc => {
                    const opt = document.createElement('option');
                    opt.value = loc; opt.textContent = loc;
                    authLocSel.appendChild(opt);
                });
            }

            inputLocality.addEventListener('change', () => {
                const auth = authorities[inputLocality.value];
                const rep = representatives[inputLocality.value];
                if (inputLocality.value !== 'Other' && (auth || rep)) {
                    authorityNameDisplay.textContent = auth || '—';
                    representativeNameDisplay.textContent = rep || '—';
                    authorityInfoDiv.classList.remove('hidden');
                } else {
                    authorityInfoDiv.classList.add('hidden');
                }
            });
        } catch (e) { console.error("Localities error", e); }
    }

    async function loadReports() {
        try {
            const res = await fetch('/api/reports');
            if (res.ok) {
                reports = await res.json();
                renderMarkers();
                const count = reports.length;
                headerStat.textContent = `${count} reports`;
                if (headerStatMobile) headerStatMobile.textContent = count;
            }
        } catch (e) { console.error("Reports error", e); }
    }

    async function loadStats() {
        try {
            const res = await fetch('/api/stats');
            if (!res.ok) return;
            const d = await res.json();
            document.getElementById('stat-total').textContent = d.total;
            document.getElementById('stat-reported').textContent = d.reported;
            document.getElementById('stat-taken-up').textContent = d.taken_up;
            document.getElementById('stat-being-solved').textContent = d.being_solved;
            document.getElementById('stat-solved').textContent = d.solved;

            const colors = ['#ef4444','#f59e0b','#facc15','#16a34a','#3b82f6','#8b5cf6','#ec4899','#06b6d4'];
            if (wasteChart) wasteChart.destroy();
            wasteChart = new Chart(document.getElementById('chart-types'), {
                type: 'doughnut',
                data: { labels: Object.keys(d.by_type), datasets: [{ data: Object.values(d.by_type), backgroundColor: colors }] },
                options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'bottom', labels: { font: { size: 11 } } } } }
            });
            if (areaChart) areaChart.destroy();
            areaChart = new Chart(document.getElementById('chart-areas'), {
                type: 'bar',
                data: { labels: Object.keys(d.by_area), datasets: [{ label: 'Reports', data: Object.values(d.by_area), backgroundColor: '#16a34a', borderRadius: 4 }] },
                options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { ticks: { font: { size: 10 } } }, y: { beginAtZero: true, ticks: { stepSize: 1 } } } }
            });

            const tbody = document.getElementById('area-table-body');
            tbody.innerHTML = '';
            Object.entries(d.by_area).sort((a, b) => b[1] - a[1]).forEach(([area, count]) => {
                const tr = document.createElement('tr');
                const auth = authorities[area] || '—';
                const rep = representatives[area] || '—';
                tr.innerHTML = `
                    <td>${area}</td>
                    <td><span style="font-size:.75rem;color:var(--text-secondary)">${auth}</span></td>
                    <td><span style="font-size:.75rem;font-style:italic;color:var(--text-secondary)">${rep}</span></td>
                    <td><strong>${count}</strong></td>
                `;
                tbody.appendChild(tr);
            });
            if (Object.keys(d.by_area).length === 0) tbody.innerHTML = '<tr><td colspan="4" style="text-align:center;color:#9ca3af;">No data yet</td></tr>';
        } catch (e) { console.error("Stats error", e); }
    }

    // ═══════ GALLERY ═══════
    function initGallery() {
        galleryAreaSelect.onchange = loadGallery;
        if (galleryAreaSelect.value) loadGallery();
    }

    async function loadGallery() {
        const area = galleryAreaSelect.value;
        galleryGrid.innerHTML = '';
        if (!area) {
            galleryGrid.innerHTML = '<div class="gallery-empty">Select an area to view reported images.</div>';
            return;
        }
        try {
            const res = await fetch(`/api/reports/area/${encodeURIComponent(area)}`);
            if (!res.ok) return;
            const data = await res.json();
            if (data.length === 0) {
                galleryGrid.innerHTML = '<div class="gallery-empty">No reports in this area yet.</div>';
                return;
            }
            data.forEach(r => {
                const card = document.createElement('div');
                card.className = 'gallery-card';
                card.onclick = () => viewReportDetail(r.id);
                const sc = r.status.toLowerCase().replace(/ /g, '-');
                card.innerHTML = `
                    ${r.image_path ? `<img src="${r.image_path}" alt="${r.waste_type}">` : '<div style="height:160px;background:#f3f4f6;"></div>'}
                    <div class="gallery-card-body">
                        <div class="gallery-card-title">${r.waste_type || 'Unknown'} <span class="status-badge ${sc}">${r.status}</span></div>
                        <div class="gallery-card-meta">📍 ${r.locality} · ${r.reported_at ? new Date(r.reported_at).toLocaleDateString() : ''}</div>
                        <div class="gallery-card-meta" style="font-size:0.75rem;color:var(--text-secondary);margin-top:2px;">📦 Items: ${r.all_detected_objects}</div>
                        <div class="gallery-card-meta" style="margin-top:6px; padding-top:4px; border-top:1px solid var(--border);">🏛️ Authority: <strong>${r.authority_name || 'Unassigned'}</strong></div>
                        <div class="gallery-card-meta" style="font-size:0.72rem; font-style:italic; color:var(--primary);">👤 Representative: <strong>${r.representative_name || 'Area Representative'}</strong></div>
                        ${r.after_image_path ? '<div class="gallery-card-meta" style="color:var(--success);font-weight:600;">✅ Resolved photo available</div>' : ''}
                    </div>
                `;
                galleryGrid.appendChild(card);
            });
        } catch (e) { console.error("Gallery error", e); }
    }

    // ═══════ REPORT FLOW ═══════
    function initReportFlow() {
        fileInput.addEventListener('change', () => { if (fileInput.files.length) handleFile(fileInput.files[0]); });
        uploadZone.addEventListener('dragover', e => e.preventDefault());
        uploadZone.addEventListener('drop', e => { e.preventDefault(); if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]); });
        btnClearPreview.addEventListener('click', () => { currentFile = null; previewContainer.classList.add('hidden'); uploadZone.classList.remove('hidden'); btnAnalyze.disabled = true; fileInput.value = ''; });
        btnAnalyze.addEventListener('click', analyzeImage);
        btnGps.addEventListener('click', gpsLocate);
        btnSubmitReport.addEventListener('click', submitReport);
        btnNewReport.addEventListener('click', resetReportFlow);
        btnGoMap.addEventListener('click', () => { resetReportFlow(); switchToView('view-map'); });
    }

    function handleFile(file) {
        if (!file.type.startsWith('image/')) return;
        currentFile = file;
        const reader = new FileReader();
        reader.onload = e => { previewImg.src = e.target.result; previewContainer.classList.remove('hidden'); uploadZone.classList.add('hidden'); btnAnalyze.disabled = false; };
        reader.readAsDataURL(file);
    }

    async function analyzeImage() {
        if (!currentFile) return;
        reportLoader.classList.remove('hidden');
        btnAnalyze.disabled = true;
        const fd = new FormData();
        fd.append('file', currentFile);
        try {
            const res = await fetch('/api/analyze', { method: 'POST', body: fd });
            const data = await res.json();
            if (res.ok && data.predictions && data.predictions.length > 0) {
                lastAnalysis = data;
                showStep('step-locate');
                resultType.textContent = data.waste_type;
                resultConfidence.textContent = `${(data.confidence * 100).toFixed(1)}%`;
                resultSeverityBadge.textContent = data.severity;
                resultSeverityBadge.className = 'result-severity-badge ' + (data.severity === 'macro-waste' ? 'macro' : 'micro');
                resultImage.onload = () => {
                    // Use SVG for robust scaling
                    const w = data.image_width || resultImage.naturalWidth || 1000;
                    const h = data.image_height || resultImage.naturalHeight || 1000;
                    console.log(`Rendering ${data.predictions.length} predictions on ${w}x${h} SVG`);
                    
                    detectionSvg.setAttribute('viewBox', `0 0 ${w} ${h}`);
                    
                    // Calculate font scale based on displayed image size
                    const displayRect = resultImage.getBoundingClientRect();
                    const displayWidth = displayRect.width || 500;
                    const fontScale = w / displayWidth;
                    const fontSize = Math.max(14 * fontScale, 14); 
                    
                    let svgContent = '';
                    data.predictions.forEach((p, idx) => {
                        const [x1, y1, x2, y2] = p.bbox;
                        const bw = x2 - x1;
                        const bh = y2 - y1;
                        const labelText = `${p.class} ${(p.confidence * 100).toFixed(0)}%`;
                        const labelWidth = labelText.length * (fontSize * 0.6);
                        
                        svgContent += `
                            <rect x="${x1}" y="${y1}" width="${bw}" height="${bh}" class="detection-box" style="stroke-width: ${2.5 * fontScale}px"></rect>
                            <rect x="${x1}" y="${y1 - (fontSize * 1.2)}" width="${labelWidth}" height="${fontSize * 1.2}" class="detection-label-bg"></rect>
                            <text x="${x1 + (fontSize * 0.2)}" y="${y1 - (fontSize * 0.3)}" class="detection-label-text" style="font-size: ${fontSize}px">${labelText}</text>
                        `;
                    });
                    detectionSvg.innerHTML = svgContent;
                };
                resultImage.src = previewImg.src;
                // Trigger onload manually if image is already cached
                if (resultImage.complete) resultImage.onload();
                detectionsList.innerHTML = '';
                data.predictions.forEach(p => {
                    const li = document.createElement('li');
                    li.innerHTML = `<span>${p.class}</span><span>${(p.confidence * 100).toFixed(1)}%</span>`;
                    detectionsList.appendChild(li);
                });
                initPickerMap();
                gpsLocate();
            } else { alert(data.message || 'No waste detected.'); }
        } catch (e) { alert('Network error.'); }
        finally { reportLoader.classList.add('hidden'); btnAnalyze.disabled = false; }
    }

    function initPickerMap() {
        if (pickerMap) pickerMap.remove();
        const lat = parseFloat(inputLat.value) || DEFAULT_LAT, lng = parseFloat(inputLng.value) || DEFAULT_LNG;
        pickerMap = L.map('location-picker-map').setView([lat, lng], 14);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 19 }).addTo(pickerMap);
        pickerMarker = L.marker([lat, lng], { draggable: true }).addTo(pickerMap);
        pickerMap.on('click', e => { pickerMarker.setLatLng(e.latlng); inputLat.value = e.latlng.lat.toFixed(4); inputLng.value = e.latlng.lng.toFixed(4); });
        pickerMarker.on('dragend', () => { const p = pickerMarker.getLatLng(); inputLat.value = p.lat.toFixed(4); inputLng.value = p.lng.toFixed(4); });
        setTimeout(() => pickerMap.invalidateSize(), 200);
    }

    function gpsLocate() {
        if (!("geolocation" in navigator)) { alert('GPS not available.'); return; }
        btnGps.classList.add('locating'); btnGps.textContent = 'Detecting...';
        navigator.geolocation.getCurrentPosition(pos => {
            inputLat.value = pos.coords.latitude.toFixed(4);
            inputLng.value = pos.coords.longitude.toFixed(4);
            if (pickerMarker) { pickerMarker.setLatLng([pos.coords.latitude, pos.coords.longitude]); pickerMap.setView([pos.coords.latitude, pos.coords.longitude], 15); }
            btnGps.classList.remove('locating');
            btnGps.innerHTML = '✓ Location Detected';
        }, err => {
            btnGps.classList.remove('locating');
            btnGps.textContent = 'Detect Location (GPS)';
            let msg = 'GPS failed.';
            if (err.code === 1) msg = 'Location permission denied. Please enable it in browser settings.';
            else if (err.code === 2) msg = 'Location unavailable (signal lost).';
            else if (err.code === 3) msg = 'Location request timed out.';
            alert(msg + ' You can still drop a pin on the map manually.');
        }, { timeout: 10000, enableHighAccuracy: true });
    }

    async function submitReport() {
        if (!lastAnalysis) return;
        const lat = parseFloat(inputLat.value), lng = parseFloat(inputLng.value), locality = inputLocality.value;
        if (!lat || !lng) { alert('Set location first.'); return; }
        btnSubmitReport.disabled = true; btnSubmitReport.textContent = 'Submitting...';
        const fd = new FormData();
        fd.append('file', currentFile); fd.append('lat', lat); fd.append('lng', lng); fd.append('locality', locality);
        if (getToken()) fd.append('token', getToken());
        try {
            const res = await fetch('/api/reports', { method: 'POST', body: fd });
            const data = await res.json();
            if (res.ok) {
                showStep('step-success');
                successMeta.innerHTML = `
                    <p><strong>Type:</strong> ${data.waste_type}</p>
                    <p><strong>Items:</strong> ${data.all_detected_objects || data.waste_type}</p>
                    <p><strong>Area:</strong> ${locality}</p>
                    <p><strong>Authority:</strong> ${data.responsible_authority || '—'}</p>
                    <p><strong>Representative:</strong> ${data.representative_name || '—'}</p>
                    <p><strong>Confidence:</strong> ${(data.confidence * 100).toFixed(1)}%</p>
                `;
                loadReports();
            } else { alert(data.detail || 'Failed'); }
        } catch (e) { alert('Network error'); }
        finally { btnSubmitReport.disabled = false; btnSubmitReport.textContent = 'Submit Report'; }
    }

    function showStep(id) {
        document.querySelectorAll('.report-step').forEach(s => s.classList.remove('active'));
        document.getElementById(id).classList.add('active');
    }
    function resetReportFlow() {
        currentFile = null; lastAnalysis = null; fileInput.value = '';
        previewContainer.classList.add('hidden'); uploadZone.classList.remove('hidden');
        btnAnalyze.disabled = true; inputLat.value = DEFAULT_LAT; inputLng.value = DEFAULT_LNG;
        inputLocality.value = 'Other'; authorityInfoDiv.classList.add('hidden');
        showStep('step-upload');
    }

    // ═══════ AUTH ═══════
    function initAuth() {
        btnAuthToggle.addEventListener('click', () => { if (!currentUser) authModal.classList.remove('hidden'); });
        modalClose.addEventListener('click', () => authModal.classList.add('hidden'));
        document.querySelector('#auth-modal .modal-backdrop').addEventListener('click', () => authModal.classList.add('hidden'));
        document.querySelectorAll('.auth-tab').forEach(tab => {
            tab.addEventListener('click', () => {
                document.querySelectorAll('.auth-tab').forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                document.querySelectorAll('#auth-modal .auth-form').forEach(f => f.classList.remove('active'));
                document.getElementById(`form-${tab.dataset.tab}`).classList.add('active');
            });
        });
        document.getElementById('login-form').addEventListener('submit', async e => {
            e.preventDefault();
            const err = document.getElementById('login-error');
            const fd = new URLSearchParams();
            fd.append('username', document.getElementById('login-email').value);
            fd.append('password', document.getElementById('login-password').value);
            try {
                const res = await fetch('/api/auth/login', { method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' }, body: fd });
                const data = await res.json();
                if (res.ok) { localStorage.setItem('token', data.access_token); currentUser = data.user; onLogin(); authModal.classList.add('hidden'); err.classList.add('hidden'); }
                else { err.textContent = data.detail; err.classList.remove('hidden'); }
            } catch (ex) { err.textContent = 'Network error'; err.classList.remove('hidden'); }
        });
        document.getElementById('register-form').addEventListener('submit', async e => {
            e.preventDefault();
            const err = document.getElementById('register-error');
            const fd = new FormData();
            fd.append('name', document.getElementById('register-name').value);
            fd.append('email', document.getElementById('register-email').value);
            fd.append('password', document.getElementById('register-password').value);
            try {
                const res = await fetch('/api/auth/register', { method: 'POST', body: fd });
                const data = await res.json();
                if (res.ok) { localStorage.setItem('token', data.access_token); currentUser = data.user; onLogin(); authModal.classList.add('hidden'); err.classList.add('hidden'); }
                else { err.textContent = data.detail; err.classList.remove('hidden'); }
            } catch (ex) { err.textContent = 'Network error'; err.classList.remove('hidden'); }
        });
        btnLogout.addEventListener('click', () => {
            localStorage.removeItem('token'); currentUser = null;
            userMenu.classList.add('hidden'); document.getElementById('auth-area').classList.remove('hidden');
            navAdmin.classList.add('hidden'); switchToView('view-map'); renderMarkers();
        });
        reportModalClose.addEventListener('click', () => reportModal.classList.add('hidden'));
        reportModalBackdrop.addEventListener('click', () => reportModal.classList.add('hidden'));
        checkAuth();
    }

    function getToken() { return localStorage.getItem('token'); }

    async function checkAuth() {
        const token = getToken(); if (!token) return;
        try {
            const res = await fetch('/api/auth/me', { headers: { 'Authorization': `Bearer ${token}` } });
            if (res.ok) { currentUser = await res.json(); onLogin(); }
        } catch (e) {}
    }

    function onLogin() {
        document.getElementById('auth-area').classList.add('hidden');
        userMenu.classList.remove('hidden');
        userEmailDisplay.textContent = currentUser.name || currentUser.email;
        userRoleBadge.textContent = currentUser.role;
        userRoleBadge.className = 'user-badge ' + currentUser.role;
        navAdmin.classList.toggle('hidden', currentUser.role !== 'admin');
        renderMarkers();
    }

    // ═══════ REPORT DETAIL MODAL ═══════
    window.viewReportDetail = async function(id) {
        try {
            const res = await fetch(`/api/reports/${id}`);
            if (!res.ok) return;
            const r = await res.json();
            renderReportDetail(r);
            reportModal.classList.remove('hidden');
        } catch (e) { console.error(e); }
    };

    function renderReportDetail(r) {
        const stageIdx = STAGES.indexOf(r.status);
        const isAdmin = currentUser && currentUser.role === 'admin';
        const isLoggedIn = !!currentUser;
        let stageHtml = '<div class="stage-labels">' + STAGES.map(s => `<span>${s}</span>`).join('') + '</div><div class="stage-progress">';
        STAGES.forEach((_, i) => {
            const cls = i < stageIdx ? 'completed' : i === stageIdx ? 'active' : '';
            stageHtml += `<div class="stage-dot ${cls}"></div>`;
            if (i < STAGES.length - 1) stageHtml += `<div class="stage-line ${i < stageIdx ? 'completed' : ''}"></div>`;
        });
        stageHtml += '</div>';

        let beforeAfterHtml = '';
        if (r.status === 'Solved' && r.after_image_path) {
            beforeAfterHtml = `<div class="before-after"><div class="before-after-col"><div class="before-after-label">🔴 Before</div>${r.image_path ? `<img src="${r.image_path}">` : ''}</div><div class="before-after-col"><div class="before-after-label">🟢 After</div><img src="${r.after_image_path}"></div></div>`;
        }

        let verifyHtml = '';
        if (r.status === 'Solved') {
            if (r.user_verified === true) verifyHtml = `<div class="verification-section"><h4>Community Verification</h4><span class="verified-badge confirmed">✅ Verified as Resolved</span>${r.verification_comment ? `<p>"${r.verification_comment}"</p>` : ''}</div>`;
            else if (isLoggedIn) verifyHtml = `<div class="verification-section"><h4>🔍 Verify Resolution</h4><textarea id="verify-comment" placeholder="Add a comment" rows="2"></textarea><div style="display:flex;gap:.5rem;"><button class="btn-primary btn-sm" onclick="verifyReport(${r.id}, true)">✅ Yes</button><button class="btn-primary btn-sm" onclick="verifyReport(${r.id}, false)">⚠️ No</button></div></div>`;
        }

        let adminHtml = '';
        if (isAdmin && r.status !== 'Solved') {
            adminHtml = `<div style="margin-top:.85rem;border-top:1px solid var(--border);padding-top:.85rem;"><h4>🛡️ Admin Actions</h4><select id="detail-status-select">${STAGES.map(s => `<option value="${s}">${s}</option>`).join('')}</select><textarea id="detail-notes" placeholder="Notes" rows="2"></textarea><input type="file" id="detail-after-photo" accept="image/*"><button class="btn-primary btn-block" onclick="updateReportStatus(${r.id})">Update Status</button></div>`;
        }

        reportDetailContent.innerHTML = `
            <div class="report-detail">
                <h3>${r.waste_type || 'Unknown'} <span class="status-badge ${r.status.toLowerCase().replace(/ /g, '-')}">${r.status}</span></h3>
                ${r.image_path && r.status !== 'Solved' ? `<img src="${r.image_path}">` : ''}
                ${beforeAfterHtml}
                ${stageHtml}
                <div class="meta-grid">
                    <div class="meta-item"><label>Area</label>${r.locality || '—'}</div>
                    <div class="meta-item"><label>Authority</label>${r.authority_name || 'Unassigned'}</div>
                    <div class="meta-item"><label>Representative</label>${r.representative_name || 'Area Representative'}</div>
                    <div class="meta-item"><label>Reported</label>${r.reported_at ? new Date(r.reported_at).toLocaleDateString() : '—'}</div>
                    <div class="meta-item"><label>Reporter</label>${r.reporter_email}</div>
                </div>
                ${verifyHtml}
                ${adminHtml}
            </div>
        `;
    }

    window.updateReportStatus = async function(id) {
        const s = document.getElementById('detail-status-select').value, n = document.getElementById('detail-notes').value, f = document.getElementById('detail-after-photo');
        const fd = new FormData(); fd.append('status', s); if (n) fd.append('notes', n); if (f.files.length) fd.append('file', f.files[0]);
        try {
            const res = await fetch(`/api/admin/reports/${id}/status`, { method: 'POST', headers: { 'Authorization': `Bearer ${getToken()}` }, body: fd });
            if (res.ok) { alert('Updated'); reportModal.classList.add('hidden'); loadReports(); loadAdminData(); }
        } catch (e) {}
    };

    window.verifyReport = async function(id, v) {
        const c = document.getElementById('verify-comment')?.value || '';
        const fd = new FormData(); fd.append('verified', v); if (c) fd.append('comment', c);
        try {
            const res = await fetch(`/api/reports/${id}/verify`, { method: 'POST', headers: { 'Authorization': `Bearer ${getToken()}` }, body: fd });
            if (res.ok) { alert('Verified'); reportModal.classList.add('hidden'); loadReports(); }
        } catch (e) {}
    };

    // ═══════ ADMIN PANEL ═══════
    async function loadAdminData() {
        if (!currentUser || currentUser.role !== 'admin') return;
        try {
            const res = await fetch('/api/admin/authorities', { headers: { 'Authorization': `Bearer ${token}` } });
            if (res.ok) {
                const auths = await res.json();
                const container = document.getElementById('authorities-list');
                container.innerHTML = auths.map(a => `<div class="authority-card"><h4>${a.organization_name}</h4><p>${a.name} — ${a.email}</p><p>📍 ${a.assigned_locality}</p></div>`).join('');
            }
        } catch (e) {}
        loadAdminReports();
    }

    async function loadAdminReports() {
        const res = await fetch('/api/reports'); if (!res.ok) return;
        const all = await res.json(), sf = document.getElementById('admin-filter-status').value;
        renderReportCards(sf ? all.filter(r => r.status === sf) : all, 'admin-reports-list');
    }

    function renderReportCards(list, containerId) {
        const container = document.getElementById(containerId);
        container.innerHTML = list.map(r => `
            <div class="report-card" onclick="viewReportDetail(${r.id})">
                ${r.image_path ? `<img class="report-thumb" src="${r.image_path}">` : '<div class="report-thumb"></div>'}
                <div class="report-card-body">
                    <div class="report-card-title">#${r.id} ${r.waste_type} <span class="status-badge ${r.status.toLowerCase().replace(/ /g, '-')}">${r.status}</span></div>
                    <div class="report-card-meta">📍 ${r.locality} · 🏛️ ${r.authority_name}</div>
                    <div class="report-card-meta" style="font-style:italic; font-size:0.7rem; color:var(--primary);">👤 Representative: ${r.representative_name || 'Area Representative'}</div>
                </div>
            </div>
        `).join('');
    }
});

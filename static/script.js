/**
 * FaceScan AI v2.0 — Frontend JavaScript
 * Three.js 3D particle animation, enhanced UI interactions,
 * animated gauges, and all API integrations
 */

// Auto-detect if running from file:// (no server) or from a Flask server
const IS_FILE_PROTOCOL = window.location.protocol === 'file:';
const API_BASE = IS_FILE_PROTOCOL ? '' : '';

// ==========================================
// THREE.JS 3D PARTICLE FACE MESH
// ==========================================

function initThreeJS() {
    const canvas = document.getElementById("heroCanvas");
    if (!canvas || typeof THREE === "undefined") return;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    // Create particles
    const particleCount = 1800;
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);
    const velocities = [];

    // Face-like distribution with neural network connections
    for (let i = 0; i < particleCount; i++) {
        const i3 = i * 3;
        
        if (i < 400) {
            // Face oval shape
            const angle = (i / 400) * Math.PI * 2;
            const rx = 2.5 + Math.random() * 0.6;
            const ry = 3.2 + Math.random() * 0.6;
            positions[i3] = Math.cos(angle) * rx + (Math.random() - 0.5) * 0.5;
            positions[i3 + 1] = Math.sin(angle) * ry + (Math.random() - 0.5) * 0.5;
            positions[i3 + 2] = (Math.random() - 0.5) * 1.5;
        } else if (i < 550) {
            // Left eye
            const angle = ((i - 400) / 150) * Math.PI * 2;
            positions[i3] = Math.cos(angle) * 0.5 - 0.9 + (Math.random() - 0.5) * 0.3;
            positions[i3 + 1] = Math.sin(angle) * 0.35 + 0.8 + (Math.random() - 0.5) * 0.2;
            positions[i3 + 2] = (Math.random() - 0.5) * 0.5 + 0.3;
        } else if (i < 700) {
            // Right eye  
            const angle = ((i - 550) / 150) * Math.PI * 2;
            positions[i3] = Math.cos(angle) * 0.5 + 0.9 + (Math.random() - 0.5) * 0.3;
            positions[i3 + 1] = Math.sin(angle) * 0.35 + 0.8 + (Math.random() - 0.5) * 0.2;
            positions[i3 + 2] = (Math.random() - 0.5) * 0.5 + 0.3;
        } else if (i < 800) {
            // Nose
            const t = (i - 700) / 100;
            positions[i3] = (Math.random() - 0.5) * 0.4;
            positions[i3 + 1] = 0.6 - t * 1.8 + (Math.random() - 0.5) * 0.2;
            positions[i3 + 2] = 0.5 + Math.sin(t * Math.PI) * 0.5;
        } else if (i < 950) {
            // Mouth
            const angle = ((i - 800) / 150) * Math.PI;
            positions[i3] = Math.cos(angle) * 1.0 + (Math.random() - 0.5) * 0.2;
            positions[i3 + 1] = -Math.sin(angle) * 0.3 - 1.2 + (Math.random() - 0.5) * 0.2;
            positions[i3 + 2] = (Math.random() - 0.5) * 0.4 + 0.2;
        } else {
            // Floating particles around
            positions[i3] = (Math.random() - 0.5) * 16;
            positions[i3 + 1] = (Math.random() - 0.5) * 12;
            positions[i3 + 2] = (Math.random() - 0.5) * 8 - 3;
        }

        // Vibrant multiple colors for white background
        const colorPalette = [
            [0.1, 0.5, 0.9], // Blue
            [0.9, 0.2, 0.4], // Pink/Red
            [0.0, 0.7, 0.5], // Teal/Green
            [0.9, 0.5, 0.1], // Orange/Yellow
            [0.6, 0.2, 0.8]  // Purple
        ];
        const c = colorPalette[Math.floor(Math.random() * colorPalette.length)];
        colors[i3] = c[0] + (Math.random() * 0.15); 
        colors[i3 + 1] = c[1] + (Math.random() * 0.15); 
        colors[i3 + 2] = c[2] + (Math.random() * 0.15);

        velocities.push({
            x: (Math.random() - 0.5) * 0.005,
            y: (Math.random() - 0.5) * 0.005,
            z: (Math.random() - 0.5) * 0.003,
        });
    }

    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));

    const material = new THREE.PointsMaterial({
        size: 0.05,
        vertexColors: true,
        transparent: true,
        opacity: 0.85,
        blending: THREE.NormalBlending,
        depthWrite: false,
    });

    const points = new THREE.Points(geometry, material);
    scene.add(points);

    // Add connection lines between nearby face particles
    const lineGeometry = new THREE.BufferGeometry();
    const linePositions = [];
    const faceParticleCount = 950;
    
    for (let i = 0; i < faceParticleCount; i += 3) {
        for (let j = i + 1; j < Math.min(i + 20, faceParticleCount); j += 2) {
            const dx = positions[i * 3] - positions[j * 3];
            const dy = positions[i * 3 + 1] - positions[j * 3 + 1];
            const dz = positions[i * 3 + 2] - positions[j * 3 + 2];
            const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
            
            if (dist < 0.8) {
                linePositions.push(
                    positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2],
                    positions[j * 3], positions[j * 3 + 1], positions[j * 3 + 2]
                );
            }
        }
    }

    lineGeometry.setAttribute("position", new THREE.Float32BufferAttribute(linePositions, 3));
    const lineMaterial = new THREE.LineBasicMaterial({
        color: 0x6c5ce7,
        transparent: true,
        opacity: 0.12,
        blending: THREE.NormalBlending,
    });
    const lines = new THREE.LineSegments(lineGeometry, lineMaterial);
    scene.add(lines);

    camera.position.z = 7;

    // Mouse interaction
    let mouseX = 0, mouseY = 0;
    document.addEventListener("mousemove", (e) => {
        mouseX = (e.clientX / window.innerWidth) * 2 - 1;
        mouseY = -(e.clientY / window.innerHeight) * 2 + 1;
    });

    // Animation loop
    let time = 0;
    function animate() {
        requestAnimationFrame(animate);
        time += 0.005;

        const posArray = geometry.attributes.position.array;
        for (let i = 0; i < particleCount; i++) {
            const i3 = i * 3;
            posArray[i3] += Math.sin(time + i * 0.01) * 0.002;
            posArray[i3 + 1] += Math.cos(time + i * 0.015) * 0.002;
        }
        geometry.attributes.position.needsUpdate = true;

        // Gentle rotation following mouse
        points.rotation.y += (mouseX * 0.3 - points.rotation.y) * 0.02;
        points.rotation.x += (-mouseY * 0.2 - points.rotation.x) * 0.02;
        points.rotation.z = Math.sin(time * 0.3) * 0.05;

        lines.rotation.copy(points.rotation);

        renderer.render(scene, camera);
    }

    animate();

    // Handle resize
    window.addEventListener("resize", () => {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    });
}

// Initialize Three.js when DOM is ready
document.addEventListener("DOMContentLoaded", () => {
    // Small delay to let Three.js CDN load
    setTimeout(initThreeJS, 100);
    initCustomCursor();
});

// Also try immediately in case DOM is already ready
if (document.readyState !== "loading") {
    setTimeout(initThreeJS, 200);
    if (!document.querySelector('.custom-cursor')) {
        initCustomCursor();
    }
}

function initCustomCursor() {
    if (document.querySelector('.custom-cursor')) return;
    const cur = document.createElement('div');
    cur.className = 'custom-cursor';
    const curDot = document.createElement('div');
    curDot.className = 'custom-cursor-dot';
    document.body.appendChild(cur);
    document.body.appendChild(curDot);
    
    let mouseX = 0, mouseY = 0;
    let dotX = 0, dotY = 0;
    
    document.addEventListener('mousemove', (e) => {
        mouseX = e.clientX;
        mouseY = e.clientY;
        curDot.style.left = mouseX + 'px';
        curDot.style.top = mouseY + 'px';
    });
    
    function animateCursor() {
        dotX += (mouseX - dotX) * 0.3;
        dotY += (mouseY - dotY) * 0.3;
        cur.style.left = dotX + 'px';
        cur.style.top = dotY + 'px';
        requestAnimationFrame(animateCursor);
    }
    animateCursor();

    const addHover = () => cur.classList.add('active');
    const removeHover = () => cur.classList.remove('active');

    document.addEventListener('mouseover', (e) => {
        if (e.target.closest('button, a, input, .upload-zone, .nav-tab, .person-card')) {
            addHover();
        } else {
            removeHover();
        }
    });
}

// ==========================================
// UTILITY FUNCTIONS
// ==========================================

function showToast(message, type = "info") {
    const container = document.getElementById("toastContainer");
    const toast = document.createElement("div");
    toast.className = `toast toast-${type}`;
    
    const icons = { success: "✅", error: "❌", info: "ℹ️" };
    toast.innerHTML = `<span class="toast-icon">${icons[type] || "ℹ️"}</span>${message}`;
    
    container.appendChild(toast);
    setTimeout(() => toast.remove(), 4200);
}

function setLoading(btn, loading) {
    const loader = btn.querySelector(".btn-loader");
    const text = btn.querySelector(".btn-text");
    if (loading) {
        loader?.classList.remove("hidden");
        text?.classList.add("hidden");
        btn.disabled = true;
    } else {
        loader?.classList.add("hidden");
        text?.classList.remove("hidden");
        btn.disabled = false;
    }
}

async function apiCall(endpoint, options = {}) {
    if (IS_FILE_PROTOCOL) {
        console.warn("Running in demo mode (file://). API calls are not available.");
        return { success: false, message: "Server not running. Please start the Flask server (python app.py) to use this feature, or open via http://localhost:5000" };
    }
    try {
        const response = await fetch(`${API_BASE}${endpoint}`, options);
        const data = await response.json();
        return data;
    } catch (error) {
        console.error("API Error:", error);
        return { success: false, message: `Network error: ${error.message}` };
    }
}

// Particle burst effect
function particleBurst(x, y, color = "#00cec9") {
    const container = document.getElementById("particleContainer");
    if (!container) return;
    
    for (let i = 0; i < 24; i++) {
        const particle = document.createElement("div");
        particle.className = "particle";
        particle.style.left = x + "px";
        particle.style.top = y + "px";
        particle.style.background = color;
        
        const angle = (Math.PI * 2 * i) / 24;
        const distance = 60 + Math.random() * 80;
        particle.style.setProperty("--dx", Math.cos(angle) * distance + "px");
        particle.style.setProperty("--dy", Math.sin(angle) * distance + "px");
        particle.style.width = (4 + Math.random() * 6) + "px";
        particle.style.height = particle.style.width;
        
        container.appendChild(particle);
        setTimeout(() => particle.remove(), 1200);
    }
}

// Animated counter
function animateCounter(element, target, duration = 1000) {
    const start = parseInt(element.textContent) || 0;
    const startTime = performance.now();
    
    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
        const current = Math.round(start + (target - start) * eased);
        element.textContent = current;
        
        if (progress < 1) requestAnimationFrame(update);
    }
    
    requestAnimationFrame(update);
}

// ==========================================
// TAB NAVIGATION
// ==========================================

document.querySelectorAll(".nav-tab").forEach(tab => {
    tab.addEventListener("click", () => {
        const targetTab = tab.dataset.tab;

        document.querySelectorAll(".nav-tab").forEach(t => t.classList.remove("active"));
        tab.classList.add("active");

        document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
        const panel = document.getElementById(`panel-${targetTab}`);
        if (panel) panel.classList.add("active");

        if (targetTab === "database") loadDatabase();

        if (targetTab !== "live" && typeof stopWebcam === "function") {
            stopWebcam();
        }
    });
});

// ==========================================
// HEALTH CHECK
// ==========================================

async function checkHealth() {
    const statusDot = document.querySelector(".status-dot");
    const statusText = document.querySelector(".status-text");

    if (IS_FILE_PROTOCOL) {
        statusDot.classList.remove("online");
        statusDot.classList.add("offline");
        statusText.textContent = "Demo Mode";
        return;
    }

    const data = await apiCall("/api/health");

    if (data.status === "ok") {
        statusDot.classList.add("online");
        statusDot.classList.remove("offline");
        statusText.textContent = `Online · ${data.known_faces} faces`;
    } else {
        statusDot.classList.add("offline");
        statusDot.classList.remove("online");
        statusText.textContent = "Offline";
    }
}

checkHealth();
if (!IS_FILE_PROTOCOL) {
    setInterval(checkHealth, 30000);
}

// ==========================================
// FILE UPLOAD HELPERS
// ==========================================

function setupUploadZone(dropZoneId, inputId, previewId, imageId, clearId, onFile) {
    const dropZone = document.getElementById(dropZoneId);
    const input = document.getElementById(inputId);
    const preview = document.getElementById(previewId);
    const image = document.getElementById(imageId);
    const clearBtn = document.getElementById(clearId);

    if (!dropZone || !input) return;

    let selectedFile = null;

    dropZone.addEventListener("click", () => input.click());

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("dragover");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    input.addEventListener("change", () => {
        if (input.files.length) {
            handleFile(input.files[0]);
        }
    });

    function handleFile(file) {
        selectedFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            if (image) image.src = e.target.result;
            if (dropZone) dropZone.classList.add("hidden");
            if (preview) preview.classList.remove("hidden");
            if (onFile) onFile(file);
        };
        reader.readAsDataURL(file);
    }

    if (clearBtn) {
        clearBtn.addEventListener("click", () => {
            selectedFile = null;
            if (input) input.value = "";
            if (image) image.src = "";
            if (dropZone) dropZone.classList.remove("hidden");
            if (preview) preview.classList.add("hidden");
            if (onFile) onFile(null);
        });
    }

    return () => selectedFile;
}

// ==========================================
// RECOGNIZE TAB
// ==========================================

let recognizeFile = null;

const getRecognizeFile = setupUploadZone(
    "recognizeDropZone", "recognizeInput", "recognizePreview",
    "recognizeImage", "recognizeClear",
    (file) => {
        recognizeFile = file;
        document.getElementById("recognizeBtn").disabled = !file;
        document.getElementById("recognizeResults").classList.add("hidden");
        const canvas = document.getElementById("recognizeCanvas");
        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Hide scan line
        const scanLine = document.getElementById("scanLine");
        if (scanLine) scanLine.classList.add("hidden");
    }
);

document.getElementById("recognizeBtn").addEventListener("click", async () => {
    if (!recognizeFile) return;

    const btn = document.getElementById("recognizeBtn");
    setLoading(btn, true);
    
    // Show scanning animation
    const scanLine = document.getElementById("scanLine");
    if (scanLine) scanLine.classList.remove("hidden");

    const formData = new FormData();
    formData.append("image", recognizeFile);

    const result = await apiCall("/api/recognize", {
        method: "POST",
        body: formData
    });

    setLoading(btn, false);
    if (scanLine) scanLine.classList.add("hidden");

    const resultsDiv = document.getElementById("recognizeResults");
    resultsDiv.classList.remove("hidden");

    if (!result.success) {
        resultsDiv.innerHTML = `
            <div class="result-card">
                <div class="result-header">
                    <span class="result-title">❌ Error</span>
                    <span class="result-badge badge-danger">Failed</span>
                </div>
                <p style="color: var(--text-secondary)">${result.message}</p>
            </div>
        `;
        showToast(result.message, "error");
        return;
    }

    drawFaceBoxes(result.faces);

    if (result.faces_found === 0) {
        resultsDiv.innerHTML = `
            <div class="result-card">
                <div class="result-header">
                    <span class="result-title">🔍 No Faces Found</span>
                    <span class="result-badge badge-warning">0 faces</span>
                </div>
                <p style="color: var(--text-secondary)">No faces were detected in this image. Try a clearer photo.</p>
            </div>
        `;
        showToast("No faces detected", "info");
        return;
    }

    // Particle burst on success
    const rect = btn.getBoundingClientRect();
    particleBurst(rect.left + rect.width / 2, rect.top, "#00cec9");

    let facesHtml = result.faces.map((face, i) => {
        const isKnown = face.name !== "Unknown";
        const initial = face.name.charAt(0).toUpperCase();
        const confClass = face.confidence >= 70 ? "high" : face.confidence >= 40 ? "medium" : "low";
        const badgeClass = isKnown ? "badge-success" : "badge-warning";
        const badgeText = isKnown ? "Matched" : "Unknown";

        return `
            <div class="face-entry">
                <div class="face-avatar">${initial}</div>
                <div class="face-info">
                    <div class="face-name">${face.name} <span style="font-size: 0.85em; color: var(--text-secondary); margin-left: 8px;">${face.expression || ''}</span></div>
                    <div class="face-meta">Face #${i + 1} · Position: (${face.location.left}, ${face.location.top})</div>
                    ${isKnown ? `
                        <div class="confidence-bar-wrapper">
                            <div class="confidence-label">
                                <span>Confidence</span>
                                <span>${face.confidence}%</span>
                            </div>
                            <div class="confidence-bar">
                                <div class="confidence-fill ${confClass}" style="width: ${face.confidence}%"></div>
                            </div>
                        </div>
                    ` : ''}
                </div>
                <span class="result-badge ${badgeClass}">${badgeText}</span>
            </div>
        `;
    }).join("");

    resultsDiv.innerHTML = `
        <div class="result-card">
            <div class="result-header">
                <span class="result-title">🎯 Recognition Results</span>
                <span class="result-badge badge-info">${result.faces_found} face(s)</span>
            </div>
            ${facesHtml}
        </div>
    `;

    showToast(`Detected ${result.faces_found} face(s)`, "success");
    checkHealth();
});

function drawFaceBoxes(faces) {
    const img = document.getElementById("recognizeImage");
    const canvas = document.getElementById("recognizeCanvas");

    const draw = () => {
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        faces.forEach(face => {
            const { top, right, bottom, left } = face.location;
            const isKnown = face.name !== "Unknown";
            const color = isKnown ? "#00cec9" : "#feca57";

            // Draw box with glow
            ctx.shadowColor = color;
            ctx.shadowBlur = 12;
            ctx.strokeStyle = color;
            ctx.lineWidth = 3;
            ctx.strokeRect(left, top, right - left, bottom - top);
            ctx.shadowBlur = 0;

            // Corner accents
            const cornerLen = 15;
            ctx.lineWidth = 4;
            ctx.strokeStyle = color;
            // Top-left
            ctx.beginPath();
            ctx.moveTo(left, top + cornerLen); ctx.lineTo(left, top); ctx.lineTo(left + cornerLen, top);
            ctx.stroke();
            // Top-right
            ctx.beginPath();
            ctx.moveTo(right - cornerLen, top); ctx.lineTo(right, top); ctx.lineTo(right, top + cornerLen);
            ctx.stroke();
            // Bottom-left
            ctx.beginPath();
            ctx.moveTo(left, bottom - cornerLen); ctx.lineTo(left, bottom); ctx.lineTo(left + cornerLen, bottom);
            ctx.stroke();
            // Bottom-right
            ctx.beginPath();
            ctx.moveTo(right - cornerLen, bottom); ctx.lineTo(right, bottom); ctx.lineTo(right, bottom - cornerLen);
            ctx.stroke();

            ctx.lineWidth = 3;

            // Draw label background
            const exp = face.expression ? "  " + face.expression : "";
            const labelText = isKnown ? `${face.name} (${face.confidence}%)${exp}` : `Unknown${exp}`;
            ctx.font = "bold 16px Inter, sans-serif";
            const textWidth = ctx.measureText(labelText).width;

            ctx.fillStyle = color;
            const labelHeight = 30;
            ctx.beginPath();
            ctx.roundRect(left, top - labelHeight - 4, textWidth + 20, labelHeight, [6, 6, 0, 0]);
            ctx.fill();

            ctx.fillStyle = "#000";
            ctx.fillText(labelText, left + 10, top - 12);
        });
    };

    if (img.complete) draw();
    else img.onload = draw;
}

// ==========================================
// REGISTER TAB
// ==========================================

let registerFile = null;

setupUploadZone(
    "registerDropZone", "registerInput", "registerPreview",
    "registerImage", "registerClear",
    (file) => {
        registerFile = file;
        updateRegisterBtn();
    }
);

const registerName = document.getElementById("registerName");
registerName.addEventListener("input", updateRegisterBtn);

function updateRegisterBtn() {
    const btn = document.getElementById("registerBtn");
    btn.disabled = !(registerFile && registerName.value.trim());
}

document.getElementById("registerBtn").addEventListener("click", async () => {
    if (!registerFile || !registerName.value.trim()) return;

    const btn = document.getElementById("registerBtn");
    setLoading(btn, true);

    const formData = new FormData();
    formData.append("image", registerFile);
    formData.append("name", registerName.value.trim());

    const result = await apiCall("/api/register", {
        method: "POST",
        body: formData
    });

    setLoading(btn, false);

    const resultsDiv = document.getElementById("registerResults");
    resultsDiv.classList.remove("hidden");

    if (result.success) {
        resultsDiv.innerHTML = `
            <div class="result-card">
                <div class="result-header">
                    <span class="result-title">✅ Registration Successful</span>
                    <span class="result-badge badge-success">Done</span>
                </div>
                <p style="color: var(--text-secondary)">${result.message}</p>
            </div>
        `;
        showToast(result.message, "success");
        
        // Particle burst celebration
        const rect = btn.getBoundingClientRect();
        particleBurst(rect.left + rect.width / 2, rect.top, "#6c5ce7");
        
        checkHealth();
    } else {
        resultsDiv.innerHTML = `
            <div class="result-card">
                <div class="result-header">
                    <span class="result-title">❌ Registration Failed</span>
                    <span class="result-badge badge-danger">Error</span>
                </div>
                <p style="color: var(--text-secondary)">${result.message}</p>
            </div>
        `;
        showToast(result.message, "error");
    }
});

// ==========================================
// COMPARE TAB (Enhanced with Gauges)
// ==========================================

let compareFile1 = null;
let compareFile2 = null;

setupUploadZone(
    "compare1DropZone", "compare1Input", "compare1Preview",
    "compare1Image", "compare1Clear",
    (file) => {
        compareFile1 = file;
        updateCompareBtn();
    }
);

setupUploadZone(
    "compare2DropZone", "compare2Input", "compare2Preview",
    "compare2Image", "compare2Clear",
    (file) => {
        compareFile2 = file;
        updateCompareBtn();
    }
);

function updateCompareBtn() {
    document.getElementById("compareBtn").disabled = !(compareFile1 && compareFile2);
}

function createGaugeSVG(value, label, color) {
    const radius = 45;
    const circumference = 2 * Math.PI * radius;
    const offset = circumference - (value / 100) * circumference;
    
    return `
        <div class="gauge-item">
            <svg class="gauge-svg" width="120" height="120" viewBox="0 0 120 120">
                <circle class="gauge-track" cx="60" cy="60" r="${radius}"/>
                <circle class="gauge-fill" cx="60" cy="60" r="${radius}" 
                    stroke="${color}" 
                    stroke-dasharray="${circumference}" 
                    stroke-dashoffset="${offset}"/>
                <text x="60" y="56" text-anchor="middle" fill="${color}" 
                    font-size="22" font-weight="800" font-family="JetBrains Mono, monospace"
                    transform="rotate(90, 60, 60)">${value}%</text>
                <text x="60" y="75" text-anchor="middle" fill="#8888aa" 
                    font-size="9" font-weight="600" 
                    transform="rotate(90, 60, 60)">${label}</text>
            </svg>
        </div>
    `;
}

document.getElementById("compareBtn").addEventListener("click", async () => {
    if (!compareFile1 || !compareFile2) return;

    const btn = document.getElementById("compareBtn");
    setLoading(btn, true);

    const formData = new FormData();
    formData.append("image1", compareFile1);
    formData.append("image2", compareFile2);

    const result = await apiCall("/api/compare", {
        method: "POST",
        body: formData
    });

    setLoading(btn, false);

    const resultsDiv = document.getElementById("compareResults");
    resultsDiv.classList.remove("hidden");

    if (!result.success) {
        resultsDiv.innerHTML = `
            <div class="result-card">
                <div class="result-header">
                    <span class="result-title">❌ Comparison Failed</span>
                    <span class="result-badge badge-danger">Error</span>
                </div>
                <p style="color: var(--text-secondary)">${result.message}</p>
            </div>
        `;
        showToast(result.message, "error");
        return;
    }

    const isMatch = result.is_same_person;
    const matchColor = isMatch ? "var(--success)" : "var(--danger)";
    const geoSim = result.geometric_similarity || result.confidence;
    const featSim = result.feature_similarity || result.confidence;

    // Particle burst on match
    if (isMatch) {
        const rect = btn.getBoundingClientRect();
        particleBurst(rect.left + rect.width / 2, rect.top, "#00cec9");
    }

    resultsDiv.innerHTML = `
        <div class="result-card">
            <div class="compare-result">
                <div class="compare-result-icon">${isMatch ? "✅" : "❌"}</div>
                <div class="compare-result-text" style="color: ${matchColor}">
                    ${isMatch ? "Same Person!" : "Different People"}
                </div>
                <div class="compare-result-detail">${result.message}</div>
                <div class="gauge-container">
                    ${createGaugeSVG(result.confidence, "Overall", isMatch ? "#00cec9" : "#ff6b6b")}
                    ${createGaugeSVG(featSim, "Feature", "#6c5ce7")}
                    ${createGaugeSVG(geoSim, "Geometric", "#74b9ff")}
                </div>
            </div>
        </div>
    `;

    showToast(result.message, isMatch ? "success" : "info");
});

// ==========================================
// DATABASE TAB (with animated counters)
// ==========================================

async function loadDatabase() {
    const dbContent = document.getElementById("dbContent");
    dbContent.innerHTML = `
        <div class="db-loading">
            <div class="skeleton-pulse"></div>
            <div class="skeleton-pulse" style="width: 40%; margin-top: 8px;"></div>
            <p style="margin-top: 16px;">Loading database...</p>
        </div>`;

    const result = await apiCall("/api/faces");

    if (!result.success) {
        dbContent.innerHTML = `<div class="db-loading">Error loading database.</div>`;
        return;
    }

    if (result.total_people === 0) {
        dbContent.innerHTML = `
            <div class="db-empty">
                <div class="db-empty-icon">👤</div>
                <div class="db-empty-text">No faces registered yet</div>
                <div class="db-empty-hint">Go to the Register tab to add your first face!</div>
            </div>
        `;
        return;
    }

    let peopleHtml = result.people.map(person => {
        const initial = person.name.charAt(0).toUpperCase();
        return `
            <div class="person-card" id="person-${person.name}">
                <div class="person-info">
                    <div class="person-avatar">${initial}</div>
                    <div>
                        <div class="person-name">${person.name}</div>
                        <div class="person-samples">${person.samples} sample(s)</div>
                    </div>
                </div>
                <button class="btn btn-danger btn-sm btn-delete-person" data-name="${person.name}">🗑 Delete</button>
            </div>
        `;
    }).join("");

    dbContent.innerHTML = `
        <div class="db-stats">
            <div class="stat-card">
                <div class="stat-value" id="statPeople">0</div>
                <div class="stat-label">People</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="statSamples">0</div>
                <div class="stat-label">Total Samples</div>
            </div>
        </div>
        <div class="people-list">
            ${peopleHtml}
        </div>
    `;

    // Animate counters
    setTimeout(() => {
        const statPeople = document.getElementById("statPeople");
        const statSamples = document.getElementById("statSamples");
        if (statPeople) animateCounter(statPeople, result.total_people, 800);
        if (statSamples) animateCounter(statSamples, result.total_faces, 800);
    }, 100);
}

async function deletePerson(name) {
    const result = await apiCall(`/api/faces/${encodeURIComponent(name)}`, {
        method: "DELETE"
    });

    if (result.success) {
        showToast(result.message, "success");
        loadDatabase();
        checkHealth();
    } else {
        showToast(result.message || "Delete failed", "error");
    }
}

// Event delegation for delete buttons (attached to document to ensure it always works)
document.addEventListener("click", (e) => {
    const btn = e.target.closest(".btn-delete-person");
    if (btn) {
        e.preventDefault();
        const personName = btn.dataset.name;
        if (personName) {
            deletePerson(personName);
        }
    }
});

document.getElementById("refreshDbBtn").addEventListener("click", loadDatabase);

document.getElementById("reloadBtn").addEventListener("click", async () => {
    const btn = document.getElementById("reloadBtn");
    btn.disabled = true;
    btn.textContent = "⏳ Rebuilding...";

    const result = await apiCall("/api/reload", { method: "POST" });

    btn.disabled = false;
    btn.textContent = "⚡ Rebuild Encodings";

    if (result.success) {
        showToast(result.message, "success");
        loadDatabase();
        checkHealth();
    } else {
        showToast(result.message || "Failed to rebuild encodings", "error");
    }
});

// ==========================================
// LIVE TAB
// ==========================================

let webcamStream = null;
let liveInterval = null;
const webcamVideo = document.getElementById("webcamVideo");
const webcamCanvas = document.getElementById("webcamCanvas");
const videoOverlay = document.getElementById("videoOverlay");
const startWebcamBtn = document.getElementById("startWebcamBtn");
const stopWebcamBtn = document.getElementById("stopWebcamBtn");
const liveResults = document.getElementById("liveResults");
const liveFacesCount = document.getElementById("liveFacesCount");
const liveFacesList = document.getElementById("liveFacesList");

startWebcamBtn?.addEventListener("click", async () => {
    try {
        webcamStream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, facingMode: "user" },
            audio: false
        });
        webcamVideo.srcObject = webcamStream;
        
        startWebcamBtn.classList.add("hidden");
        stopWebcamBtn.classList.remove("hidden");
        videoOverlay.classList.add("hidden");
        liveResults.classList.remove("hidden");
        
        webcamVideo.onloadedmetadata = () => {
            webcamCanvas.width = webcamVideo.videoWidth;
            webcamCanvas.height = webcamVideo.videoHeight;
        };

        liveInterval = setInterval(processWebcamFrame, 600);
        showToast("Webcam started", "success");
    } catch (err) {
        console.error("Error accessing webcam:", err);
        showToast("Could not access webcam. Please check permissions.", "error");
    }
});

stopWebcamBtn?.addEventListener("click", stopWebcam);

function stopWebcam() {
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
    }
    if (liveInterval) {
        clearInterval(liveInterval);
        liveInterval = null;
    }
    
    webcamVideo.srcObject = null;
    startWebcamBtn?.classList.remove("hidden");
    stopWebcamBtn?.classList.add("hidden");
    videoOverlay?.classList.remove("hidden");
    liveResults?.classList.add("hidden");
    
    if (webcamCanvas) {
        const ctx = webcamCanvas.getContext("2d");
        ctx.clearRect(0, 0, webcamCanvas.width, webcamCanvas.height);
    }
}

async function processWebcamFrame() {
    if (!webcamVideo || webcamVideo.paused || webcamVideo.ended) return;

    if (!document.getElementById("panel-live").classList.contains("active")) {
        return;
    }

    const offCanvas = document.createElement("canvas");
    offCanvas.width = webcamVideo.videoWidth;
    offCanvas.height = webcamVideo.videoHeight;
    const offCtx = offCanvas.getContext("2d");
    
    offCtx.drawImage(webcamVideo, 0, 0, offCanvas.width, offCanvas.height);
    
    const base64Data = offCanvas.toDataURL("image/jpeg", 0.7);

    try {
        const result = await apiCall("/api/recognize_base64", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image_base64: base64Data })
        });

        if (result && result.success) {
            drawLiveFaces(result.faces);
            updateLiveStats(result.faces);
        } else {
            const ctx = webcamCanvas.getContext("2d");
            ctx.clearRect(0, 0, webcamCanvas.width, webcamCanvas.height);
        }
    } catch (error) {
        console.error("Error processing webcam frame:", error);
    }
}

function drawLiveFaces(faces) {
    const ctx = webcamCanvas.getContext("2d");
    ctx.clearRect(0, 0, webcamCanvas.width, webcamCanvas.height);

    const w = webcamCanvas.width;

    faces.forEach(face => {
        const { top, right, bottom, left } = face.location;
        const isKnown = face.name !== "Unknown";
        const color = isKnown ? "#00cec9" : "#feca57";

        const mirroredLeft = w - right;
        const mirroredRight = w - left;

        // Glow effect
        ctx.shadowColor = color;
        ctx.shadowBlur = 12;
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(mirroredLeft, top, mirroredRight - mirroredLeft, bottom - top);
        ctx.shadowBlur = 0;

        // Corner accents
        const cornerLen = 12;
        ctx.lineWidth = 4;
        // Top-left
        ctx.beginPath();
        ctx.moveTo(mirroredLeft, top + cornerLen); ctx.lineTo(mirroredLeft, top); ctx.lineTo(mirroredLeft + cornerLen, top);
        ctx.stroke();
        // Top-right
        ctx.beginPath();
        ctx.moveTo(mirroredRight - cornerLen, top); ctx.lineTo(mirroredRight, top); ctx.lineTo(mirroredRight, top + cornerLen);
        ctx.stroke();
        // Bottom-left
        ctx.beginPath();
        ctx.moveTo(mirroredLeft, bottom - cornerLen); ctx.lineTo(mirroredLeft, bottom); ctx.lineTo(mirroredLeft + cornerLen, bottom);
        ctx.stroke();
        // Bottom-right
        ctx.beginPath();
        ctx.moveTo(mirroredRight - cornerLen, bottom); ctx.lineTo(mirroredRight, bottom); ctx.lineTo(mirroredRight, bottom - cornerLen);
        ctx.stroke();

        ctx.lineWidth = 3;

        // Draw label
        const exp = face.expression ? "  " + face.expression : "";
        const labelText = isKnown ? `${face.name} (${face.confidence}%)${exp}` : `Unknown${exp}`;
        ctx.font = "bold 15px Inter, sans-serif";
        const textWidth = ctx.measureText(labelText).width;

        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.roundRect(mirroredLeft, top - 30, textWidth + 18, 28, [6, 6, 0, 0]);
        ctx.fill();

        ctx.fillStyle = "#000";
        ctx.fillText(labelText, mirroredLeft + 9, top - 10);
    });
}

function updateLiveStats(faces) {
    liveFacesCount.textContent = `${faces.length} faces`;
    
    if (faces.length === 0) {
        liveFacesList.innerHTML = `<p style="color: var(--text-muted); padding: 12px 0;">No faces detected in view.</p>`;
        return;
    }

    let facesHtml = faces.map((face, i) => {
        const isKnown = face.name !== "Unknown";
        const initial = face.name.charAt(0).toUpperCase();
        const confClass = face.confidence >= 70 ? "high" : face.confidence >= 40 ? "medium" : "low";
        const badgeClass = isKnown ? "badge-success" : "badge-warning";
        const badgeText = isKnown ? "Matched" : "Unknown";

        return `
            <div class="face-entry" style="padding: 12px; margin-bottom: 8px;">
                <div class="face-avatar" style="width: 38px; height: 38px; font-size: 1rem;">${initial}</div>
                <div class="face-info" style="font-size: 0.9rem;">
                    <div class="face-name">
                        ${face.name}
                        <span style="font-size: 0.85em; color: var(--text-secondary); margin-left: 4px;">${face.expression || ''}</span>
                    </div>
                    ${isKnown ? `
                        <div class="confidence-bar-wrapper" style="margin-top: 5px;">
                            <div class="confidence-bar" style="height: 6px;">
                                <div class="confidence-fill ${confClass}" style="width: ${face.confidence}%"></div>
                            </div>
                        </div>
                    ` : ''}
                </div>
                <span class="result-badge ${badgeClass}" style="font-size: 0.65rem;">${badgeText}</span>
            </div>
        `;
    }).join("");

    liveFacesList.innerHTML = facesHtml;
}

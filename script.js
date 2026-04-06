async function uploadImage() {
    console.log("uploadImage called");

    const fileInput = document.getElementById("fileInput");
    const conf = document.getElementById("confSlider").value;

    if (!fileInput.files.length) {
        alert("Válassz ki egy képet!");
        return;
    }

    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(`${API_URL}/predict?conf=${conf}`, {
        method: "POST",
        body: formData
    });

    if (!response.ok) {
        alert("Hiba történt az elemzés során.");
        return;
    }

    const data = await response.json();

    drawResult(file, data.detections);
    showRecommendations(data.recommendations);
    showPlan(data.plan);
}

// ===============================
// BACKEND API URL (Railway)
// ===============================
const API_URL = "https://web-production-75595.up.railway.app";

const COLORS = ["#f97316", "#22c55e", "#3b82f6", "#eab308", "#ec4899", "#a855f7", "#f43f5e"];

// ===============================
// KÉPFELTÖLTÉS ÉS PREDIKCIÓ
// ===============================
async function uploadImage() {
    const fileInput = document.getElementById("fileInput");
    const conf = document.getElementById("confSlider").value;

    if (!fileInput.files.length) {
        alert("Válassz ki egy képet!");
        return;
    }

    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append("file", file);

    // 🔥 Itt javítottuk a fetch-et → Railway backendre megy
    const response = await fetch(`${API_URL}/predict?conf=${conf}`, {
        method: "POST",
        body: formData
    });

    if (!response.ok) {
        alert("Hiba történt az elemzés során.");
        return;
    }

    const data = await response.json();

    drawResult(file, data.detections);
    showRecommendations(data.recommendations);
    showPlan(data.plan);
}


// ===============================
// TISZTA KÉP + TÖBB METSZÉSI PONT (CLUSTERELVE)
// ===============================
function drawResult(file, detections) {
    const canvas = document.getElementById("resultCanvas");
    const ctx = canvas.getContext("2d");

    const img = new Image();
    img.onload = () => {
        canvas.width = img.width;
        canvas.height = img.height;

        ctx.drawImage(img, 0, 0);

        const cutPoints = detections
            .filter(det => det.class === 0)
            .map(det => {
                const [x1, y1, x2, y2] = det.bbox;
                return {
                    x: (x1 + x2) / 2,
                    y: (y1 + y2) / 2,
                    bbox: det.bbox,
                    confidence: det.confidence
                };
            });

        if (cutPoints.length === 0) return;

        const merged = [];
        const threshold = 80;

        cutPoints.forEach(p => {
            let foundCluster = false;

            for (let m of merged) {
                const dx = m.x - p.x;
                const dy = m.y - p.y;
                const dist = Math.sqrt(dx * dx + dy * dy);

                if (dist < threshold) {
                    if (p.confidence > m.confidence) {
                        m.x = p.x;
                        m.y = p.y;
                        m.bbox = p.bbox;
                        m.confidence = p.confidence;
                    }
                    foundCluster = true;
                    break;
                }
            }

            if (!foundCluster) {
                merged.push({ ...p });
            }
        });

        merged.forEach(p => {
            drawCutGuide(ctx, p.bbox);
        });
    };

    img.src = URL.createObjectURL(file);
}


// ===============================
// METSZÉSI ÚTMUTATÓ FUNKCIÓK
// ===============================
function drawCutPoint(ctx, x, y) {
    ctx.fillStyle = "#ff4444";
    ctx.font = "28px system-ui";
    ctx.fillText("✂️", x - 10, y - 10);
}

function getBranchAngle(bbox) {
    const [x1, y1, x2, y2] = bbox;
    const dx = x2 - x1;
    const dy = y2 - y1;
    return Math.atan2(dy, dx);
}

function drawCutDirection(ctx, x, y, angle) {
    const length = 40;
    const x2 = x + Math.cos(angle) * length;
    const y2 = y + Math.sin(angle) * length;

    ctx.strokeStyle = "#ff4444";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x2, y2);
    ctx.stroke();
}

function drawBudDistance(ctx, x, y, angle) {
    const offset = 15;
    const x2 = x + Math.cos(angle) * offset;
    const y2 = y + Math.sin(angle) * offset;

    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.arc(x2, y2, 4, 0, Math.PI * 2);
    ctx.fill();
}

function drawCutGuide(ctx, bbox) {
    const [x1, y1, x2, y2] = bbox;
    const x = (x1 + x2) / 2;
    const y = (y1 + y2) / 2;

    const angle = getBranchAngle(bbox);

    drawCutPoint(ctx, x, y);
    drawCutDirection(ctx, x, y, angle);
    drawBudDistance(ctx, x, y, angle);
}


// ===============================
// METSZÉSI AJÁNLATOK MEGJELENÍTÉSE
// ===============================
function showRecommendations(recommendations) {
    const container = document.getElementById("recommendations-container");
    if (!container) return;
    container.innerHTML = "";

    recommendations.forEach(rec => {
        const div = document.createElement("div");
        div.className = `recommendation-item priority-${rec.priority}`;

        let icon = "🌿";
        if (rec.type === "beteg ág") icon = "⚠️";
        if (rec.type === "elhalt ág") icon = "🪵";
        if (rec.type === "vízhajtás") icon = "💧";
        if (rec.type === "vastag ág") icon = "🪚";
        if (rec.type === "keresztező ág") icon = "🔀";
        if (rec.type === "metszési pont") icon = "✂️";

        div.innerHTML = `
            <span class="icon">${icon}</span>
            <div>
                <strong>${rec.type}</strong><br>
                ${rec.advice}
            </div>
        `;

        container.appendChild(div);
    });
}


// ===============================
// INTELLIGENS METSZÉSI TERV MEGJELENÍTÉSE
// ===============================
function showPlan(plan) {
    const container = document.getElementById("plan-container");
    if (!container) return;
    container.innerHTML = "";

    if (!plan || plan.length === 0) {
        container.innerHTML = "<p>Nincs elegendő információ a metszési sorrendhez.</p>";
        return;
    }

    plan.forEach(step => {
        const div = document.createElement("div");
        div.className = "plan-item";

        const typeLabel = step.type || "Ág";

        div.innerHTML = `
            <strong>${step.step}. lépés – ${typeLabel}</strong><br>
            ${step.reason}
        `;

        container.appendChild(div);
    });
}


// ===============================
// SLIDER ÉRTÉK KIÍRÁSA
// ===============================
document.getElementById("confSlider").oninput = function () {
    document.getElementById("confValue").innerText = parseFloat(this.value).toFixed(2);
};


// ===============================
// FÁJLKEZELÉS, DRAG & DROP, KAMERA, COOKIE
// ===============================
function triggerFileInput() {
    document.getElementById("fileInput").click();
}

const dropzone = document.getElementById("dropzone");
const previewImage = document.getElementById("previewImage");

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("dragover", e => {
    e.preventDefault();
    dropzone.classList.add("dragover");
});
dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
});
dropzone.addEventListener("drop", e => {
    e.preventDefault();
    dropzone.classList.remove("dragover");

    const file = e.dataTransfer.files[0];
    if (!file) return;

    const dt = new DataTransfer();
    dt.items.add(file);
    fileInput.files = dt.files;

    showPreview(file);
});

fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) {
        showPreview(fileInput.files[0]);
    }
});

function showPreview(file) {
    const reader = new FileReader();
    reader.onload = e => {
        previewImage.src = e.target.result;
        previewImage.style.display = "block";
    };
    reader.readAsDataURL(file);
}

let cameraStream = null;

async function openCamera() {
    const container = document.getElementById("cameraContainer");
    const preview = document.getElementById("cameraPreview");

    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "environment" }
        });

        preview.srcObject = cameraStream;
        container.style.display = "block";

    } catch (err) {
        alert("A kamera nem érhető el vagy nincs engedélyezve.");
    }
}

function takePhoto() {
    const video = document.getElementById("cameraPreview");
    const canvas = document.createElement("canvas");

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0);

    canvas.toBlob(blob => {
        const file = new File([blob], "camera_photo.jpg", { type: "image/jpeg" });
        const dt = new DataTransfer();
        dt.items.add(file);
        fileInput.files = dt.files;

        showPreview(file);
        stopCamera();
    }, "image/jpeg", 0.95);
}

function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }
    document.getElementById("cameraContainer").style.display = "none";
}

function showCookieBanner() {
    if (!localStorage.getItem("treeai_cookies")) {
        document.getElementById("cookieBanner").style.display = "block";
    }
}

function acceptCookies() {
    localStorage.setItem("treeai_cookies", "accepted");
    document.getElementById("cookieBanner").style.display = "none";
}

function declineCookies() {
    localStorage.setItem("treeai_cookies", "declined");
    document.getElementById("cookieBanner").style.display = "none";
}

window.addEventListener("load", showCookieBanner);

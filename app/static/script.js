/* ========================================================= */
/* DEVELOPMENT CACHE-BUSTER (SAFE VERSION) */
/* ========================================================= */
(function() {
    const currentScript = document.currentScript;
    if (!currentScript.src.includes("v=")) {
        const version = Date.now();
        const newScript = document.createElement("script");
        newScript.src = `/static/script.js?v=${version}`;
        document.body.appendChild(newScript);
        return; // <-- SAFE EXIT (no error thrown)
    }
})();

/* ========================================================= */
/* UPLOAD CSV */
/* ========================================================= */

function uploadFile() {
    const fileInput = document.getElementById("fileInput").files[0];
    if (!fileInput) {
        document.getElementById("status").innerText = "Please select a file first.";
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput);

    document.getElementById("status").innerText = "Uploading...";

    fetch("/upload", { method: "POST", body: formData })
        .then(r => r.json())
        .then(d => {
            document.getElementById("status").innerText = d.message;
            highlightStep("step-upload");
        })
        .catch(err => {
            console.error("Upload error:", err);
            document.getElementById("status").innerText = "Upload failed.";
        });
}


/* ========================================================= */
/* RUN PIPELINE */
/* ========================================================= */

function runPipeline() {
    document.getElementById("status").innerText = "Running pipeline...";
    resetFlowHighlights();

    highlightStep("step-upload");
    highlightStep("step-ingestion");

    fetch("/run", { method: "POST" })
        .then(r => r.json())
        .then(d => {
            console.log("Pipeline response:", d);

            document.getElementById("status").innerText = d.message;
            document.getElementById("download").innerHTML =
                `<a href="/download">Download Output File</a>`;

            highlightStep("step-preprocessing");
            highlightStep("step-features");
            highlightStep("step-model");
            highlightStep("step-forecast");
            highlightStep("step-evaluation");
            highlightStep("step-output");

            if (d.metrics) {
                updateMetrics(d.metrics);
            } else {
                console.warn("No metrics returned from backend.");
            }
        })
        .catch(err => {
            console.error("Pipeline error:", err);
            document.getElementById("status").innerText = "Pipeline failed.";
        });
}


/* ========================================================= */
/* FLOWCHART LOGIC */
/* ========================================================= */

function resetFlowHighlights() {
    const steps = document.querySelectorAll(".flow-step");
    steps.forEach(s => s.classList.remove("completed"));
}

function highlightStep(id) {
    const el = document.getElementById(id);
    if (el) el.classList.add("completed");
}

function stepClick(step) {
    const info = {
        upload: "Upload CSV file with historical load and related features.",
        ingestion: "Ingestion: read CSV, basic validation, and loading into DataFrame.",
        preprocessing: "Preprocessing: cleaning, handling missing values, transformations.",
        features: "Feature Engineering: time-based features, lags, calendar, etc.",
        model: "Model Training: fitting the forecasting model.",
        forecast: "Forecasting: generating predictions for the target horizon.",
        evaluation: "Evaluation: computing accuracy, reliability, and business metrics.",
        output: "Save & Output: exporting forecast results as CSV."
    };
    document.getElementById("step-info").innerText = info[step] || "";
}


/* ========================================================= */
/* COLOR CODING RULES */
/* ========================================================= */

function getColorClass(metricName, value) {
    if (value === null || isNaN(value)) return "";

    if (["mae", "rmse", "mape", "smape", "wape", "md_ae", "nrmse"].includes(metricName)) {
        if (value < 5) return "green";
        if (value < 15) return "yellow";
        return "red";
    }

    if (metricName === "r2") {
        if (value > 0.90) return "green";
        if (value > 0.70) return "yellow";
        return "red";
    }

    if (["mbe", "pbias"].includes(metricName)) {
        if (Math.abs(value) < 1) return "green";
        if (Math.abs(value) < 5) return "yellow";
        return "red";
    }

    return "";
}


/* ========================================================= */
/* METRIC VALUE UPDATER */
/* ========================================================= */

function updateMetric(id, value, metricName = null) {
    const el = document.getElementById(id);
    if (!el) return;

    if (value === null || value === undefined || value === "" || isNaN(value)) {
        el.innerHTML = `<span class="metric-value">—</span><div class="metric-color"></div>`;
        return;
    }

    const num = Number(value);
    const color = metricName ? getColorClass(metricName, num) : "";

    el.innerHTML = `
        <span class="metric-value">${num.toFixed(3)}</span>
        <div class="metric-color ${color}"></div>
    `;
}


/* ========================================================= */
/* UPDATE ALL METRICS */
/* ========================================================= */

function updateMetrics(m) {
    console.log("Updating metrics:", m);

    updateMetric("acc-mae", m.accuracy.mae, "mae");
    updateMetric("acc-rmse", m.accuracy.rmse, "rmse");
    updateMetric("acc-mape", m.accuracy.mape, "mape");
    updateMetric("acc-smape", m.accuracy.smape, "smape");
    updateMetric("acc-md_ae", m.accuracy.md_ae, "md_ae");
    updateMetric("acc-r2", m.accuracy.r2, "r2");
    updateMetric("acc-nrmse", m.accuracy.nrmse, "nrmse");
    updateMetric("acc-wape", m.accuracy.wape, "wape");

    updateMetric("rel-mbe", m.reliability.mbe, "mbe");
    updateMetric("rel-pbias", m.reliability.pbias, "pbias");
    updateMetric("rel-error_variance", m.reliability.error_variance);

    updateMetric("eff-train", m.efficiency.training_time_sec);
    updateMetric("eff-inf", m.efficiency.inference_time_sec);
    updateMetric("eff-size", m.efficiency.model_size_mb);

    const horizonDiv = document.getElementById("horizon-container");
    horizonDiv.innerHTML = "";
    Object.entries(m.horizon.mae_per_horizon).forEach(([h, v]) => {
        const num = Number(v);
        const color = getColorClass("mae", num);

        const row = document.createElement("div");
        row.className = "metric-row";
        row.innerHTML = `
            <span>H${h}</span>
            <span class="metric-value">${num.toFixed(3)}</span>
            <div class="metric-color ${color}"></div>
        `;
        horizonDiv.appendChild(row);
    });

    updateMetric("drift-status", m.drift.prediction_drift_status);

    updateMetric("biz-peak", m.business.peak_demand_mae, "mae");
    updateMetric("biz-ewm", m.business.energy_weighted_mape, "mape");
}


/* ========================================================= */
/* DOWNLOAD METRICS REPORT (CSV) */
/* ========================================================= */

function downloadMetricsReport() {
    try {

        const fullForms = {
            "MAE": "Mean Absolute Error",
            "RMSE": "Root Mean Square Error",
            "MAPE": "Mean Absolute Percentage Error",
            "SMAPE": "Symmetric Mean Absolute Percentage Error",
            "MD_AE": "Median Absolute Error",
            "R²": "Coefficient of Determination",
            "NRMSE": "Normalized Root Mean Square Error",
            "WAPE": "Weighted Absolute Percentage Error",
            "MBE": "Mean Bias Error",
            "PBIAS": "Percent Bias",
            "Error Variance": "Variance of Forecast Errors",
            "Training Time (s)": "Model Training Time (seconds)",
            "Inference Time (s)": "Model Inference Time (seconds)",
            "Model Size (MB)": "Model File Size (MB)",
            "Peak Demand MAE": "Peak Demand Mean Absolute Error",
            "Energy Weighted MAPE": "Energy Weighted Mean Absolute Percentage Error",
            "Drift Status": "Prediction Drift Status"
        };

        const metricGroups = [
            {
                category: "Accuracy",
                items: [
                    ["MAE", "acc-mae"],
                    ["RMSE", "acc-rmse"],
                    ["MAPE", "acc-mape"],
                    ["SMAPE", "acc-smape"],
                    ["MD_AE", "acc-md_ae"],
                    ["R²", "acc-r2"],
                    ["NRMSE", "acc-nrmse"],
                    ["WAPE", "acc-wape"]
                ]
            },
            {
                category: "Reliability",
                items: [
                    ["MBE", "rel-mbe"],
                    ["PBIAS", "rel-pbias"],
                    ["Error Variance", "rel-error_variance"]
                ]
            },
            {
                category: "Efficiency",
                items: [
                    ["Training Time (s)", "eff-train"],
                    ["Inference Time (s)", "eff-inf"],
                    ["Model Size (MB)", "eff-size"]
                ]
            },
            {
                category: "Business",
                items: [
                    ["Peak Demand MAE", "biz-peak"],
                    ["Energy Weighted MAPE", "biz-ewm"]
                ]
            },
            {
                category: "Drift",
                items: [
                    ["Drift Status", "drift-status"]
                ]
            }
        ];

        let csv = "Category,Metric,Value,Color,Full Form\n";

        // Process grouped metrics
        metricGroups.forEach(group => {
            group.items.forEach(([name, id]) => {
                const el = document.getElementById(id);
                if (!el) return;

                const value = el.innerText.trim();

                const row = el.closest(".metric-row");
                let color = "";
                if (row) {
                    const colorBox = row.querySelector(".metric-color");
                    if (colorBox) {
                        if (colorBox.classList.contains("green")) color = "green";
                        else if (colorBox.classList.contains("yellow")) color = "yellow";
                        else if (colorBox.classList.contains("red")) color = "red";
                    }
                }

                const full = fullForms[name] || "";

                csv += `"${group.category}","${name}","${value}","${color}","${full}"\n`;
            });
        });

        // Horizon metrics
        const horizonContainer = document.getElementById("horizon-container");
        if (horizonContainer) {
            const rows = horizonContainer.querySelectorAll(".metric-row");
            rows.forEach(row => {
                const name = row.children[0].innerText.trim();
                const value = row.children[1].innerText.trim();

                let color = "";
                const colorBox = row.children[2];
                if (colorBox.classList.contains("green")) color = "green";
                else if (colorBox.classList.contains("yellow")) color = "yellow";
                else if (colorBox.classList.contains("red")) color = "red";

                csv += `"Horizon","${name}","${value}","${color}","Horizon MAE"\n`;
            });
        }

        const blob = new Blob([csv], { type: "text/csv" });
        const url = URL.createObjectURL(blob);

        const a = document.createElement("a");
        a.href = url;
        a.download = "metrics_report.csv";
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);

        URL.revokeObjectURL(url);

    } catch (err) {
        console.error("CSV generation failed:", err);
        alert("Failed to generate CSV. Check console for details.");
    }
}

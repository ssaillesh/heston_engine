const API_BASE = '';

const elements = {
    kappa: document.getElementById('kappa'),
    theta: document.getElementById('theta'),
    sigma: document.getElementById('sigma'),
    rho: document.getElementById('rho'),
    r: document.getElementById('r'),
    q: document.getElementById('q'),
    S0: document.getElementById('S0'),
    V0: document.getElementById('V0'),
    K: document.getElementById('K'),
    T: document.getElementById('T'),
    optionType: document.getElementById('option-type'),
    fellerIndicator: document.getElementById('feller-indicator'),
    fellerRatio: document.getElementById('feller-ratio'),
    btnPrice: document.getElementById('btn-price'),
    btnGreeks: document.getElementById('btn-greeks'),
    btnSurface: document.getElementById('btn-surface'),
    btnPaths: document.getElementById('btn-paths'),
    btnCompare: document.getElementById('btn-compare'),
    btnValidate: document.getElementById('btn-validate'),
    priceValue: document.getElementById('price-value'),
    priceDetails: document.getElementById('price-details'),
    pricingResults: document.getElementById('pricing-results'),
    greeksResults: document.getElementById('greeks-results'),
    compareResults: document.getElementById('compare-results'),
    validationResults: document.getElementById('validation-results'),
    surfaceChart: document.getElementById('surface-chart'),
    pathsChart: document.getElementById('paths-chart'),
    varianceChart: document.getElementById('variance-chart')
};

function getParams() {
    return {
        kappa: parseFloat(elements.kappa.value),
        theta: parseFloat(elements.theta.value),
        sigma: parseFloat(elements.sigma.value),
        rho: parseFloat(elements.rho.value),
        r: parseFloat(elements.r.value),
        q: parseFloat(elements.q.value),
        S0: parseFloat(elements.S0.value),
        V0: parseFloat(elements.V0.value)
    };
}

async function apiRequest(endpoint, method = 'GET', data = null) {
    const options = { method, headers: { 'Content-Type': 'application/json' } };
    if (data) options.body = JSON.stringify(data);
    const response = await fetch(`${API_BASE}${endpoint}`, options);
    if (!response.ok) {
        let message = 'API request failed';
        try {
            const error = await response.json();
            message = error.error || error.message || message;
        } catch (_) {
            const text = await response.text();
            if (text) message = text;
        }
        throw new Error(message);
    }
    return response.json();
}

function formatNumber(num, decimals = 4) {
    if (typeof num !== 'number' || !Number.isFinite(num)) return '--';
    return num.toFixed(decimals);
}

function updateFellerIndicator() {
    const params = getParams();
    const ratio = (2 * params.kappa * params.theta) / (params.sigma ** 2);
    elements.fellerRatio.textContent = `2κθ/σ² = ${formatNumber(ratio, 2)}`;
    elements.fellerIndicator.classList.remove('feller-ok', 'feller-warn', 'feller-fail');
    if (!Number.isFinite(ratio)) {
        elements.fellerIndicator.classList.add('feller-fail');
        elements.fellerIndicator.querySelector('.feller-icon').textContent = '✗';
        elements.fellerIndicator.querySelector('.feller-text').textContent = 'Invalid Feller Ratio';
    } else if (ratio > 1) {
        elements.fellerIndicator.classList.add('feller-ok');
        elements.fellerIndicator.querySelector('.feller-icon').textContent = '✓';
        elements.fellerIndicator.querySelector('.feller-text').textContent = 'Feller Condition Satisfied';
    } else if (ratio > 0.5) {
        elements.fellerIndicator.classList.add('feller-warn');
        elements.fellerIndicator.querySelector('.feller-icon').textContent = '⚠';
        elements.fellerIndicator.querySelector('.feller-text').textContent = 'Feller Condition Marginal';
    } else {
        elements.fellerIndicator.classList.add('feller-fail');
        elements.fellerIndicator.querySelector('.feller-icon').textContent = '✗';
        elements.fellerIndicator.querySelector('.feller-text').textContent = 'Feller Condition Violated';
    }
}

function showResultSection(sectionId) {
    elements.pricingResults.classList.add('hidden');
    elements.greeksResults.classList.add('hidden');
    elements.compareResults.classList.add('hidden');
    elements.validationResults.classList.add('hidden');
    const section = document.getElementById(sectionId);
    if (section) section.classList.remove('hidden');
}

async function priceOption(method = 'analytical') {
    try {
        elements.btnPrice.classList.add('loading');
        const data = {
            method,
            params: getParams(),
            K: parseFloat(elements.K.value),
            T: parseFloat(elements.T.value),
            option_type: elements.optionType.value
        };
        const result = await apiRequest('/api/price', 'POST', data);
        showResultSection('pricing-results');
        elements.priceValue.textContent = formatNumber(result.price, 4);
        let details = `Method: ${result.method}`;
        if (result.stderr) {
            details += ` | Std Error: ${formatNumber(result.stderr, 6)}`;
            details += ` | 95% CI: [${formatNumber(result.confidence_95[0], 4)}, ${formatNumber(result.confidence_95[1], 4)}]`;
        }
        if (result.grid_size) details += ` | Grid: ${result.grid_size}`;
        elements.priceDetails.textContent = details;
    } catch (error) {
        alert(`Error: ${error.message}`);
    } finally {
        elements.btnPrice.classList.remove('loading');
    }
}

async function calculateGreeks() {
    try {
        elements.btnGreeks.classList.add('loading');
        const data = {
            params: getParams(),
            K: parseFloat(elements.K.value),
            T: parseFloat(elements.T.value),
            option_type: elements.optionType.value
        };
        const result = await apiRequest('/api/greeks', 'POST', data);
        showResultSection('greeks-results');
        document.getElementById('greek-delta').textContent = formatNumber(result.delta, 4);
        document.getElementById('greek-gamma').textContent = formatNumber(result.gamma, 6);
        document.getElementById('greek-vega').textContent = formatNumber(result.vega, 4);
        document.getElementById('greek-theta').textContent = formatNumber(result.theta_daily, 4);
        document.getElementById('greek-rho').textContent = formatNumber(result.rho, 4);
        document.getElementById('greek-vanna').textContent = formatNumber(result.vanna, 6);
        document.getElementById('greek-volga').textContent = formatNumber(result.volga, 4);
    } catch (error) {
        alert(`Error: ${error.message}`);
    } finally {
        elements.btnGreeks.classList.remove('loading');
    }
}

async function generateSurface() {
    try {
        elements.btnSurface.classList.add('loading');
        const params = getParams();
        const data = {
            params,
            K_min: params.S0 * 0.7,
            K_max: params.S0 * 1.3,
            n_strikes: 20,
            T_min: 0.1,
            T_max: 2.0,
            n_maturities: 15
        };
        const result = await apiRequest('/api/surface', 'POST', data);
        const plotData = [{
            type: 'surface',
            x: result.strikes,
            y: result.maturities,
            z: result.surface,
            colorscale: [[0, '#2d0a4e'], [0.25, '#622a87'], [0.5, '#a855f7'], [0.75, '#c084fc'], [1, '#f5d0fe']],
            contours: { z: { show: true, usecolormap: true, highlightcolor: '#fff', project: { z: true } } }
        }];
        const layout = {
            title: 'Heston Model Implied Volatility Surface',
            scene: {
                xaxis: { title: 'Strike (K)' },
                yaxis: { title: 'Maturity (T)' },
                zaxis: { title: 'Implied Vol (σ)' },
                camera: { eye: { x: 1.5, y: -1.5, z: 0.8 } }
            },
            paper_bgcolor: '#161b22',
            plot_bgcolor: '#161b22',
            font: { color: '#c9d1d9' },
            margin: { t: 50, b: 40, l: 40, r: 40 }
        };
        Plotly.newPlot(elements.surfaceChart, plotData, layout, { responsive: true });
        switchTab('surface-chart');
    } catch (error) {
        alert(`Error: ${error.message}`);
    } finally {
        elements.btnSurface.classList.remove('loading');
    }
}

async function simulatePaths() {
    try {
        elements.btnPaths.classList.add('loading');
        const data = {
            params: getParams(),
            T: parseFloat(elements.T.value),
            n_steps: 252,
            n_paths: 112000
        };
        const result = await apiRequest('/api/paths', 'POST', data);
        const traces = result.S_paths.map((path, i) => ({
            type: 'scatter',
            mode: 'lines',
            x: result.times,
            y: path,
            name: `Path ${i + 1}`,
            line: { width: 0.5, color: 'rgba(168, 85, 247, 0.3)' },
            showlegend: false,
            hoverinfo: 'skip'
        }));
        // Calculate average path
        const avgPath = result.times.map((_, timeIdx) => {
            const sum = result.S_paths.reduce((acc, path) => acc + path[timeIdx], 0);
            return sum / result.S_paths.length;
        });
        const avgFinalValue = avgPath[avgPath.length - 1];
        // Add average path trace
        traces.push({
            type: 'scatter',
            mode: 'lines',
            x: result.times,
            y: avgPath,
            name: 'Average Path',
            line: { width: 3, color: '#00ff88' },
            showlegend: true
        });
        const layout = {
            title: 'Monte Carlo Stock Price Paths (QE Scheme)',
            xaxis: { title: 'Time (years)', gridcolor: '#30363d' },
            yaxis: { title: 'Stock Price', gridcolor: '#30363d' },
            paper_bgcolor: '#161b22',
            plot_bgcolor: '#21262d',
            font: { color: '#c9d1d9' },
            margin: { t: 50, b: 50, l: 60, r: 80 },
            annotations: [{
                x: result.times[result.times.length - 1],
                y: avgFinalValue,
                xanchor: 'left',
                yanchor: 'middle',
                text: `Avg: ${avgFinalValue.toFixed(2)}`,
                showarrow: false,
                font: { color: '#00ff88', size: 12, family: 'monospace' },
                xshift: 5
            }],
            legend: { x: 0.02, y: 0.98, bgcolor: 'rgba(22, 27, 34, 0.8)' }
        };
        Plotly.newPlot(elements.pathsChart, traces, layout, { responsive: true });
        const varTraces = result.V_paths.map((path, i) => ({
            type: 'scatter',
            mode: 'lines',
            x: result.times,
            y: path,
            name: `Path ${i + 1}`,
            line: { width: 0.5, color: 'rgba(0, 212, 255, 0.3)' },
            showlegend: false,
            hoverinfo: 'skip'
        }));
        // Calculate average variance path
        const avgVarPath = result.times.map((_, timeIdx) => {
            const sum = result.V_paths.reduce((acc, path) => acc + path[timeIdx], 0);
            return sum / result.V_paths.length;
        });
        const avgVarFinalValue = avgVarPath[avgVarPath.length - 1];
        // Add average variance path trace
        varTraces.push({
            type: 'scatter',
            mode: 'lines',
            x: result.times,
            y: avgVarPath,
            name: 'Average Variance',
            line: { width: 3, color: '#00ff88' },
            showlegend: true
        });
        const varLayout = {
            title: 'Variance Process Paths (CIR Process)',
            xaxis: { title: 'Time (years)', gridcolor: '#30363d' },
            yaxis: { title: 'Variance', gridcolor: '#30363d' },
            paper_bgcolor: '#161b22',
            plot_bgcolor: '#21262d',
            font: { color: '#c9d1d9' },
            margin: { t: 50, b: 50, l: 60, r: 80 },
            annotations: [{
                x: result.times[result.times.length - 1],
                y: avgVarFinalValue,
                xanchor: 'left',
                yanchor: 'middle',
                text: `Avg: ${avgVarFinalValue.toFixed(4)}`,
                showarrow: false,
                font: { color: '#00ff88', size: 12, family: 'monospace' },
                xshift: 5
            }],
            legend: { x: 0.02, y: 0.98, bgcolor: 'rgba(22, 27, 34, 0.8)' }
        };
        Plotly.newPlot(elements.varianceChart, varTraces, varLayout, { responsive: true });
        switchTab('paths-chart');
    } catch (error) {
        alert(`Error: ${error.message}`);
    } finally {
        elements.btnPaths.classList.remove('loading');
    }
}

async function compareMethods() {
    try {
        elements.btnCompare.classList.add('loading');
        const data = {
            params: getParams(),
            K: parseFloat(elements.K.value),
            T: parseFloat(elements.T.value)
        };
        const result = await apiRequest('/api/compare', 'POST', data);
        showResultSection('compare-results');
        const tbody = document.getElementById('compare-tbody');
        tbody.innerHTML = `
            <tr>
                <td>Analytical (Fourier)</td>
                <td>${formatNumber(result.analytical.price, 6)}</td>
                <td>${formatNumber(result.analytical.time_ms, 1)}</td>
                <td>—</td>
            </tr>
            <tr>
                <td>PDE (ADI)</td>
                <td>${formatNumber(result.pde.price, 6)}</td>
                <td>${formatNumber(result.pde.time_ms, 1)}</td>
                <td>${formatNumber(result.differences.pde_vs_analytical, 6)} (${formatNumber(result.differences.pde_vs_analytical_pct, 3)}%)</td>
            </tr>
            <tr>
                <td>Monte Carlo (QE)</td>
                <td>${formatNumber(result.monte_carlo.price, 6)} ± ${formatNumber(result.monte_carlo.stderr, 6)}</td>
                <td>${formatNumber(result.monte_carlo.time_ms, 1)}</td>
                <td>${formatNumber(result.differences.mc_vs_analytical, 6)} (${formatNumber(result.differences.mc_vs_analytical_pct, 3)}%)</td>
            </tr>
        `;
    } catch (error) {
        alert(`Error: ${error.message}`);
    } finally {
        elements.btnCompare.classList.remove('loading');
    }
}

async function validateModel() {
    try {
        elements.btnValidate.classList.add('loading');
        const data = {
            params: getParams(),
            K: parseFloat(elements.K.value),
            T: parseFloat(elements.T.value)
        };
        const result = await apiRequest('/api/validate', 'POST', data);
        showResultSection('validation-results');
        const content = document.getElementById('validation-content');
        content.innerHTML = `
            <div class="validation-item ${result.put_call_parity.passed ? 'passed' : 'failed'}">
                <span class="validation-status">${result.put_call_parity.passed ? '✓' : '✗'}</span>
                <span class="validation-name">Put-Call Parity</span>
                <span class="validation-detail">
                    C - P = ${formatNumber(result.put_call_parity.actual, 4)} | 
                    Theoretical = ${formatNumber(result.put_call_parity.theoretical, 4)} | 
                    Error = ${formatNumber(result.put_call_parity.error, 6)}
                </span>
            </div>
            <div class="validation-item ${result.feller_condition.satisfied ? 'passed' : 'failed'}">
                <span class="validation-status">${result.feller_condition.satisfied ? '✓' : '✗'}</span>
                <span class="validation-name">Feller Condition</span>
                <span class="validation-detail">
                    2κθ = ${formatNumber(result.feller_condition.two_kappa_theta, 4)} | 
                    σ² = ${formatNumber(result.feller_condition.sigma_squared, 4)} | 
                    Ratio = ${formatNumber(result.feller_condition.ratio, 2)}
                </span>
            </div>
            <div class="validation-item">
                <span class="validation-status">ℹ</span>
                <span class="validation-name">Black-Scholes Comparison</span>
                <span class="validation-detail">
                    Heston = ${formatNumber(result.bs_comparison.heston_price, 4)} | 
                    BS = ${formatNumber(result.bs_comparison.bs_price, 4)} | 
                    Diff = ${formatNumber(result.bs_comparison.difference_pct, 2)}%
                </span>
            </div>
        `;
    } catch (error) {
        alert(`Error: ${error.message}`);
    } finally {
        elements.btnValidate.classList.remove('loading');
    }
}

function switchTab(tabId) {
    document.querySelectorAll('.viz-tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.tab === tabId);
    });
    document.querySelectorAll('.chart-container').forEach(chart => {
        chart.classList.toggle('active', chart.id === tabId);
    });
}

function setActiveButton(activeBtn) {
    const allBtns = [elements.btnPrice, elements.btnGreeks, elements.btnSurface, elements.btnPaths, elements.btnCompare, elements.btnValidate];
    allBtns.forEach(btn => btn.classList.remove('active'));
    activeBtn.classList.add('active');
}

[elements.kappa, elements.theta, elements.sigma].forEach(el => {
    el.addEventListener('input', updateFellerIndicator);
});

elements.btnPrice.addEventListener('click', () => { setActiveButton(elements.btnPrice); priceOption('analytical'); });
elements.btnGreeks.addEventListener('click', () => { setActiveButton(elements.btnGreeks); calculateGreeks(); });
elements.btnSurface.addEventListener('click', () => { setActiveButton(elements.btnSurface); generateSurface(); });
elements.btnPaths.addEventListener('click', () => { setActiveButton(elements.btnPaths); simulatePaths(); });
elements.btnCompare.addEventListener('click', () => { setActiveButton(elements.btnCompare); compareMethods(); });
elements.btnValidate.addEventListener('click', () => { setActiveButton(elements.btnValidate); validateModel(); });

document.querySelectorAll('.viz-tab').forEach(tab => {
    tab.addEventListener('click', () => switchTab(tab.dataset.tab));
});

updateFellerIndicator();
apiRequest('/api/health').then(r => console.log('API:', r)).catch(e => console.error('API Error:', e));


// LOAD PARAMETERS FROM MARKET DATA PAGE

function loadParamsFromHash() {
    if (window.location.hash && window.location.hash.length > 1) {
        try {
            var hashParams = new URLSearchParams(window.location.hash.substring(1));
            if (hashParams.has("S0") && hashParams.has("kappa")) {
                ["S0", "V0", "kappa", "theta", "sigma", "rho", "r", "q"].forEach(function(field) {
                    var value = hashParams.get(field);
                    var el = document.getElementById(field);
                    if (el && value) el.value = parseFloat(value).toFixed(field === "S0" ? 2 : 4);
                });
                var S0 = hashParams.get("S0");
                if (S0) {
                    var kField = document.getElementById("K");
                    if (kField) kField.value = Math.round(parseFloat(S0));
                }
                window.history.replaceState({}, document.title, "/");
                updateFellerIndicator();
                showNotification("Parameters loaded from Market Data", "success");
            }
        } catch (e) { console.error("Failed to parse params:", e); }
    }
}

function showNotification(message, type) {
    var n = document.createElement("div");
    n.style.cssText = "position:fixed;top:20px;right:20px;background:" + (type === "success" ? "#00ff88" : "#00d4ff") + ";color:#1e1e2e;padding:12px 20px;border-radius:8px;font-weight:bold;z-index:1000;";
    n.innerHTML = message + " <button onclick='this.parentElement.remove()' style='background:none;border:none;cursor:pointer;font-size:16px;'>x</button>";
    document.body.appendChild(n);
    setTimeout(function() { n.remove(); }, 3000);
}

loadParamsFromHash();

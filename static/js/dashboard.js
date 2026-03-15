/* ═══════════════════════════════════════════════════════
   RL Portfolio Dashboard — Premium Dark Theme JS
   Chart.js v4.4.1 — Glassmorphism Style
   Supports: SDELP-DDPG + IQL-BL model switching
   ═══════════════════════════════════════════════════════ */

/* ── Current Model State ── */
let currentModel = 'sdelp';
const MODEL_NAMES = {
    sdelp: 'SDELP-DDPG',
    iqlbl: 'IQL-BL',
};
const MODEL_COLORS = {
    sdelp: { primary: '#4f8eff', glow: 'rgba(79, 142, 255, 0.25)' },
    iqlbl: { primary: '#22d3ee', glow: 'rgba(34, 211, 238, 0.25)' },
};

/* ── Color Palette (Dark Theme) ── */
const COLORS = {
    blue: '#4f8eff',
    blueGlow: 'rgba(79, 142, 255, 0.25)',
    green: '#00e5a0',
    greenGlow: 'rgba(0, 229, 160, 0.2)',
    red: '#ff4d6a',
    redGlow: 'rgba(255, 77, 106, 0.2)',
    yellow: '#ffb020',
    purple: '#a78bfa',
    cyan: '#22d3ee',
    gray: '#5a6a80',
    grayDim: '#1e293b',
    gridLine: 'rgba(255, 255, 255, 0.06)',
    gridLineSub: 'rgba(255, 255, 255, 0.02)',
    textPrimary: '#eef2ff',
    textSecondary: '#a0aec0',
    textMuted: '#5a6a80',
    tooltipBg: 'rgba(8, 12, 24, 0.96)',
};

const CRYPTO_COLORS = [
    '#F7931A', // BTC - orange
    '#627EEA', // ETH - blue/purple
    '#C3A634', // DOGE - gold
    '#345D9D', // LTC - dark blue
    '#26A17B', // USDT - teal
    '#67B2E8', // XEM - sky blue
    '#14B6E7', // XLM - cyan
    '#9945FF', // SOL - purple
    '#0088CC', // XRP - blue
];

let charts = {};
let portfolioData = null;
let trainingHistoryData = null;

/* ── Chart.js Global Defaults (Dark Theme) ── */
Chart.defaults.color = COLORS.textSecondary;
Chart.defaults.font.family = "'Inter', 'Manrope', sans-serif";
Chart.defaults.font.size = 11;
Chart.defaults.plugins.legend.labels.usePointStyle = true;
Chart.defaults.plugins.legend.labels.padding = 16;
Chart.defaults.plugins.tooltip.backgroundColor = COLORS.tooltipBg;
Chart.defaults.plugins.tooltip.titleFont = { size: 13, weight: '600' };
Chart.defaults.plugins.tooltip.bodyFont = { size: 12 };
Chart.defaults.plugins.tooltip.padding = 14;
Chart.defaults.plugins.tooltip.cornerRadius = 10;
Chart.defaults.plugins.tooltip.borderColor = 'rgba(255,255,255,0.08)';
Chart.defaults.plugins.tooltip.borderWidth = 1;
Chart.defaults.scale.grid.color = COLORS.gridLine;

/* ── Tab Navigation ── */
let currentTab = 'dashboard';
document.addEventListener('DOMContentLoaded', () => {
    // Tab switching
    document.querySelectorAll('.nav-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            const tabName = tab.dataset.tab;
            if (tabName === currentTab) return;
            currentTab = tabName;

            // Update active tab button
            document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');

            // Toggle tab-content panels
            document.querySelectorAll('.tab-content').forEach(panel => {
                if (panel.dataset.tabContent === tabName) {
                    panel.classList.add('active');
                } else {
                    panel.classList.remove('active');
                }
            });

            // Re-render charts for newly visible tab (canvas needs to be visible)
            if (tabName === 'allocation' && portfolioData && !portfolioData.error) {
                setTimeout(() => {
                    renderWeightHistory(portfolioData);
                    renderWeightsDonutAlloc(portfolioData);
                    renderWeightsBarChart(portfolioData);
                    renderAllocKPIs(portfolioData);
                    renderWeightRanking(portfolioData);
                }, 50);
            }
            if (tabName === 'model' && trainingHistoryData) {
                setTimeout(() => {
                    renderEpisodeRewardChart(trainingHistoryData);
                    renderEpisodeValueChart(trainingHistoryData);
                    renderActorLossChart(trainingHistoryData);
                    renderCriticLossChart(trainingHistoryData);
                }, 50);
            }
        });
    });

    loadDashboard();

    // Auto-refresh every 5 minutes, aligned to clock (test mode)
    function updateLastRefreshBadge() {
        const badge = document.getElementById('badge-last-refresh');
        if (badge) {
            const now = new Date();
            const hh = String(now.getHours()).padStart(2, '0');
            const mm = String(now.getMinutes()).padStart(2, '0');
            const ss = String(now.getSeconds()).padStart(2, '0');
            badge.textContent = `Updated ${hh}:${mm}:${ss}`;
        }
    }
    updateLastRefreshBadge();

    function scheduleNext5Min() {
        const now = new Date();
        const next = new Date(now);
        next.setMinutes(Math.ceil(now.getMinutes() / 5) * 5, 0, 0);
        if (next <= now) next.setMinutes(next.getMinutes() + 5);
        setTimeout(() => {
            fetch('/api/refresh').then(() => {
                loadDashboard();
                updateLastRefreshBadge();
            });
            scheduleNext5Min();
        }, next - now);
    }
    scheduleNext5Min();
});

/* ── Load Dashboard Data ── */
async function loadDashboard() {
    showLoading(true);
    updateModelLabels();
    try {
        const [portfolio, modelInfo, trainPortfolio, trainingHistory, live5min] = await Promise.all([
            fetchJSON('/api/portfolio'),
            fetchJSON('/api/model-info'),
            fetchJSON('/api/train-portfolio'),
            fetchJSON('/api/training-history'),
            fetchJSON('/api/live-5min'),
        ]);

        portfolioData = portfolio;

        if (!portfolio.error) {
            renderMetricCards(portfolio);
            renderPortfolioChart(portfolio);
            renderDailyReturns(portfolio);
            renderMetricsTable(portfolio);
        }

        if (!live5min.error) {
            renderLive5minChart(live5min);
        }

        if (!trainPortfolio.error) {
            renderTrainPortfolioChart(trainPortfolio);
            renderTrainMetrics(trainPortfolio);
        }

        if (modelInfo) {
            if (currentModel === 'iqlbl') {
                renderIQLBLModelInfo(modelInfo);
                renderIQLBLModelInfoMini(modelInfo);
            } else {
                renderModelInfo(modelInfo);
                renderModelInfoMini(modelInfo);
            }
            const testBadge = document.getElementById('badge-test-period');
            const trainBadge = document.getElementById('badge-train-period');
            if (testBadge) testBadge.textContent = modelInfo.test_period;
            if (trainBadge) trainBadge.textContent = modelInfo.train_period;
        }

        if (trainingHistory && !trainingHistory.error) {
            trainingHistoryData = trainingHistory;
            renderEpisodeRewardChart(trainingHistory);
            renderEpisodeValueChart(trainingHistory);
            renderActorLossChart(trainingHistory);
            renderCriticLossChart(trainingHistory);
        }

        updateTimestamp();
    } catch (err) {
        console.error('Dashboard load error:', err);
    }
    showLoading(false);
}

async function fetchJSON(url) {
    // Auto-append model parameter for model-aware endpoints
    const modelAwareEndpoints = ['/api/portfolio', '/api/train-portfolio', '/api/live-5min',
                                  '/api/metrics', '/api/model-info', '/api/training-history'];
    const isModelAware = modelAwareEndpoints.some(ep => url.startsWith(ep));
    if (isModelAware) {
        const sep = url.includes('?') ? '&' : '?';
        url = `${url}${sep}model=${currentModel}`;
    }
    const res = await fetch(url);
    return res.json();
}

/* ── Model Switching ── */
function switchModel(modelId) {
    if (modelId === currentModel) return;
    currentModel = modelId;

    // Update UI labels
    updateModelLabels();

    // Clear cache and reload
    fetch('/api/refresh').then(() => loadDashboard());
}

function updateModelLabels() {
    const name = MODEL_NAMES[currentModel] || currentModel.toUpperCase();

    // Update badges
    const badgeModelParams = document.getElementById('badge-model-params');
    if (badgeModelParams) badgeModelParams.textContent = name;

    const badgeFullConfig = document.getElementById('badge-full-config');
    if (badgeFullConfig) badgeFullConfig.textContent = name;

    const badgeDailyReturns = document.getElementById('badge-daily-returns');
    if (badgeDailyReturns) badgeDailyReturns.textContent = name;

    const badgeTrainVs = document.getElementById('badge-train-vs');
    if (badgeTrainVs) badgeTrainVs.textContent = `${name} vs B&H`;

    const badgeTestVs = document.getElementById('badge-test-vs');
    if (badgeTestVs) badgeTestVs.textContent = `${name} vs B&H`;

    // Update table headers
    const thTrainModel = document.getElementById('th-train-model');
    if (thTrainModel) thTrainModel.textContent = name;

    const thTestModel = document.getElementById('th-test-model');
    if (thTestModel) thTestModel.textContent = name;

    // Update header subtitle
    const subtitle = document.querySelector('.header-subtitle');
    if (subtitle) {
        if (currentModel === 'iqlbl') {
            subtitle.textContent = 'IQL + Dynamic Black-Litterman • Daily Monitor';
        } else {
            subtitle.textContent = 'Crypto Portfolio Optimization • Daily Monitor';
        }
    }

    // Show/hide SDELP-specific Model tab sections (training charts)
    const trainingChartSections = document.querySelectorAll('[data-tab-content="model"] .grid-2, [data-tab-content="model"] .training-img-container');
    trainingChartSections.forEach(el => {
        const parentCard = el.closest('section');
        if (parentCard && parentCard !== parentCard.parentElement.firstElementChild) {
            // Hide training chart sections for IQLBL (no episode-based training)
            if (currentModel === 'iqlbl') {
                parentCard.style.display = 'none';
            } else {
                parentCard.style.display = '';
            }
        }
    });
}

function updateTimestamp() {
    const el = document.getElementById('update-time');
    if (el) {
        const now = new Date();
        el.innerHTML = `<span class="status-dot live"></span>Last Sync: ${now.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' })} KST`;
    }
}

function showLoading(show) {
    document.querySelectorAll('.loading-overlay').forEach(el => {
        el.classList.toggle('hidden', !show);
    });
}

function getMetric(metrics, possibleKeys, defaultVal = 0) {
    if (!metrics) return defaultVal;
    for (const key of possibleKeys) {
        if (metrics[key] !== undefined) return metrics[key];
    }
    return defaultVal;
}

/* ── KPI Metric Cards ── */
function renderMetricCards(portfolio) {
    // Portfolio CRR
    const portfolioEl = document.getElementById('metric-portfolio');
    if (portfolio && portfolio.sdelp_metrics) {
        let crr = getMetric(portfolio.sdelp_metrics, ['CRR', 'Cumulative Return']);
        if (crr === 0 && portfolio.sdelp_values && portfolio.sdelp_values.length > 0) {
            crr = portfolio.sdelp_values[portfolio.sdelp_values.length - 1] / portfolio.sdelp_values[0];
        }
        setMetricCard(portfolioEl, crr.toFixed(4),
            crr >= 1 ? `+${((crr - 1) * 100).toFixed(1)}%` : `${((crr - 1) * 100).toFixed(1)}%`,
            crr >= 1
        );
    }

    // Annualized Return
    const dailyEl = document.getElementById('metric-daily');
    if (portfolio && portfolio.sdelp_metrics) {
        let ar = getMetric(portfolio.sdelp_metrics, ['AR (%)', 'Annualized Return (%)', 'Annualized Return']);
        // If ar is raw (like 0.1716 instead of 17.16), multiply by 100 IF standard AR
        if (portfolio.sdelp_metrics['Annualized Return'] !== undefined && portfolio.sdelp_metrics['AR (%)'] === undefined) {
            ar = ar * 100;
        }
        setMetricCard(dailyEl, ar.toFixed(2) + '%',
            ar >= 0 ? 'Positive' : 'Negative',
            ar >= 0
        );
    }

    // Sharpe
    const sharpeEl = document.getElementById('metric-sharpe');
    if (portfolio && portfolio.sdelp_metrics) {
        const sharpe = getMetric(portfolio.sdelp_metrics, ['Sharpe', 'Sharpe Ratio']);
        setMetricCard(sharpeEl, sharpe.toFixed(3),
            sharpe >= 1 ? 'Good' : sharpe >= 0 ? 'Moderate' : 'Poor',
            sharpe >= 0
        );
    }

    // MDD
    const mddEl = document.getElementById('metric-mdd');
    if (portfolio && portfolio.sdelp_metrics) {
        let mdd = getMetric(portfolio.sdelp_metrics, ['MDD (%)', 'Max Drawdown (%)', 'MDD']);
        if (portfolio.sdelp_metrics['MDD'] !== undefined && portfolio.sdelp_metrics['MDD (%)'] === undefined) {
            mdd = Math.abs(mdd) * 100;
        }
        setMetricCard(mddEl, '-' + mdd.toFixed(2) + '%',
            mdd <= 20 ? 'Low Risk' : mdd <= 40 ? 'Medium' : 'High Risk',
            mdd <= 30
        );
    }
}

function setMetricCard(container, value, subtitle, isPositive) {
    if (!container) return;
    const valEl = container.querySelector('.metric-card-value');
    const chgEl = container.querySelector('.metric-card-change');
    if (valEl) valEl.textContent = value;
    if (chgEl && subtitle) {
        const cls = isPositive ? 'text-green' : 'text-red';
        chgEl.innerHTML = `<span class="${cls}">${subtitle}</span>`;
    }
}

/* ── Test Period Portfolio Chart ── */
function renderPortfolioChart(data) {
    const ctx = document.getElementById('portfolioChart');
    if (!ctx) return;
    if (charts.portfolio) charts.portfolio.destroy();

    const mc = MODEL_COLORS[currentModel] || MODEL_COLORS.sdelp;
    const gradientBlue = ctx.getContext('2d').createLinearGradient(0, 0, 0, 280);
    gradientBlue.addColorStop(0, mc.glow.replace('0.25', '0.20'));
    gradientBlue.addColorStop(0.5, mc.glow.replace('0.25', '0.06'));
    gradientBlue.addColorStop(1, mc.glow.replace('0.25', '0.0'));

    charts.portfolio = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.dates,
            datasets: [
                {
                    label: MODEL_NAMES[currentModel] || 'Model',
                    data: data.sdelp_values,
                    borderColor: mc.primary,
                    backgroundColor: gradientBlue,
                    borderWidth: 2.5,
                    fill: true,
                    pointRadius: 0,
                    pointHoverRadius: 5,
                    pointHoverBackgroundColor: mc.primary,
                    pointHoverBorderColor: '#fff',
                    pointHoverBorderWidth: 2,
                    tension: 0.2,
                },
                {
                    label: 'Buy & Hold',
                    data: data.bah_values,
                    borderColor: COLORS.yellow,
                    borderWidth: 1.8,
                    borderDash: [6, 4],
                    fill: false,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    tension: 0.2,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: { position: 'top' },
                tooltip: {
                    callbacks: {
                        label: ctx => `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(4)}`,
                    },
                },
            },
            scales: {
                x: {
                    type: 'time',
                    time: { unit: 'month', displayFormats: { month: 'yyyy-MM' } },
                    grid: { display: false },
                    ticks: { maxTicksLimit: 12 },
                    border: { color: COLORS.gridLine },
                },
                y: {
                    grid: { color: COLORS.gridLine },
                    ticks: { padding: 8 },
                    border: { display: false },
                },
            },
        },
    });
}

/* ── Live 5-min Portfolio Return ── */
function renderLive5minChart(data) {
    const ctx = document.getElementById('recent30dChart');
    if (!ctx) return;
    if (charts.recent30d) charts.recent30d.destroy();

    const timestamps = data.dates;
    const sdelpPct = data.sdelp_values;
    const bahPct = data.bah_values;

    const finalReturn = sdelpPct[sdelpPct.length - 1];

    // Update badge with time range
    const badge = document.getElementById('badge-30d-return');
    if (badge) {
        badge.textContent = `${timestamps[0]} ~ ${timestamps[timestamps.length - 1]}`;
        badge.className = 'badge badge-blue';
    }

    const gradientGreen = ctx.getContext('2d').createLinearGradient(0, 0, 0, 220);
    gradientGreen.addColorStop(0, finalReturn >= 0 ? 'rgba(0, 229, 160, 0.25)' : 'rgba(255, 77, 106, 0.25)');
    gradientGreen.addColorStop(1, 'rgba(0, 0, 0, 0)');

    charts.recent30d = new Chart(ctx, {
        type: 'line',
        data: {
            labels: timestamps,
            datasets: [
                {
                    label: MODEL_NAMES[currentModel] || 'Model',
                    data: sdelpPct,
                    borderColor: finalReturn >= 0 ? COLORS.green : COLORS.red,
                    backgroundColor: gradientGreen,
                    borderWidth: 2.5,
                    fill: true,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    tension: 0.3,
                },
                {
                    label: 'Buy & Hold',
                    data: bahPct,
                    borderColor: COLORS.gray,
                    borderWidth: 1.2,
                    borderDash: [4, 3],
                    fill: false,
                    pointRadius: 0,
                    tension: 0.3,
                },
                {
                    label: 'Zero',
                    data: timestamps.map(() => 0),
                    borderColor: 'rgba(255,255,255,0.15)',
                    borderWidth: 1,
                    borderDash: [2, 2],
                    fill: false,
                    pointRadius: 0,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: ctx => {
                            if (ctx.dataset.label === 'Zero') return null;
                            const sign = ctx.parsed.y >= 0 ? '+' : '';
                            return `${ctx.dataset.label}: ${sign}${ctx.parsed.y.toFixed(3)}%`;
                        },
                    },
                },
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { maxTicksLimit: 8, font: { size: 9 } },
                    border: { color: COLORS.gridLine },
                },
                y: {
                    grid: { color: COLORS.gridLine },
                    ticks: {
                        callback: v => (v >= 0 ? '+' : '') + v.toFixed(0) + '%',
                        font: { size: 9 },
                        padding: 4,
                    },
                    border: { display: false },
                },
            },
        },
    });
}

/* ── Daily Returns ── */
function renderDailyReturns(data) {
    const ctx = document.getElementById('dailyReturnsChart');
    if (!ctx) return;
    if (charts.dailyReturns) charts.dailyReturns.destroy();

    const step = Math.max(1, Math.floor(data.daily_returns.length / 120));
    const dates = data.daily_return_dates.filter((_, i) => i % step === 0);
    const returns = data.daily_returns.filter((_, i) => i % step === 0);

    const bgColors = returns.map(r => r >= 0 ? COLORS.greenGlow : COLORS.redGlow);
    const borderColors = returns.map(r => r >= 0 ? COLORS.green : COLORS.red);

    charts.dailyReturns = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: dates,
            datasets: [{
                label: 'Daily Return',
                data: returns.map(r => r * 100),
                backgroundColor: bgColors,
                borderColor: borderColors,
                borderWidth: 1,
                barPercentage: 1.0,
                categoryPercentage: 1.0,
                borderRadius: 1,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: ctx => `${ctx.parsed.y >= 0 ? '+' : ''}${ctx.parsed.y.toFixed(2)}%`,
                    },
                },
            },
            scales: {
                x: {
                    type: 'time',
                    time: { unit: 'month', displayFormats: { month: 'yyyy-MM' } },
                    grid: { display: false },
                    ticks: { maxTicksLimit: 8 },
                    border: { color: COLORS.gridLine },
                },
                y: {
                    grid: { color: COLORS.gridLine },
                    ticks: { callback: v => v.toFixed(1) + '%', padding: 8 },
                    border: { display: false },
                },
            },
        },
    });
}

/* ── Weight History Stacked ── */
function renderWeightHistory(data) {
    const ctx = document.getElementById('weightHistoryChart');
    if (!ctx) return;
    if (charts.weightHistory) charts.weightHistory.destroy();

    const labels = data.weight_dates;
    const allLabels = data.weight_labels;
    const weights = data.weights_sampled;
    const colors = ['#475569', ...CRYPTO_COLORS.slice(0, allLabels.length - 1)];

    const datasets = allLabels.map((name, i) => ({
        label: name,
        data: weights.map(w => w[i] * 100),
        backgroundColor: hexToRgba(colors[i], 0.6),
        borderColor: colors[i],
        borderWidth: 1,
        fill: true,
        pointRadius: 0,
    }));

    charts.weightHistory = new Chart(ctx, {
        type: 'line',
        data: { labels, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: {
                    position: 'top',
                    labels: { padding: 10, font: { size: 10 } },
                },
                tooltip: {
                    callbacks: {
                        label: ctx => `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(1)}%`,
                    },
                },
            },
            scales: {
                x: {
                    type: 'time',
                    time: { unit: 'month', displayFormats: { month: 'yyyy-MM' } },
                    grid: { display: false },
                    ticks: { maxTicksLimit: 10 },
                    border: { color: COLORS.gridLine },
                },
                y: {
                    stacked: true,
                    max: 100,
                    min: 0,
                    grid: { color: COLORS.gridLine },
                    ticks: { callback: v => v + '%', padding: 8 },
                    border: { display: false },
                },
            },
        },
    });
}

/* ── Metrics Table ── */
function renderMetricsTable(data) {
    const tbody = document.getElementById('metrics-tbody');
    if (!tbody) return;

    const m1 = data.sdelp_metrics;
    const m2 = data.bah_metrics;

    const metrics = [
        { key: 'CRR', label: 'CRR (Cumulative Return)', fmt: 4, higherBetter: true },
        { key: 'AR (%)', label: 'Annualized Return (%)', fmt: 2, higherBetter: true },
        { key: 'Sharpe', label: 'Sharpe Ratio', fmt: 3, higherBetter: true },
        { key: 'Sortino', label: 'Sortino Ratio', fmt: 3, higherBetter: true },
        { key: 'AV (%)', label: 'Annualized Volatility (%)', fmt: 2, higherBetter: false },
        { key: 'MDD (%)', label: 'Max Drawdown (%)', fmt: 2, higherBetter: false },
    ];

    tbody.innerHTML = metrics.map(m => {
        const v1 = m1[m.key];
        const v2 = m2[m.key];
        const sdelp_wins = m.higherBetter ? v1 > v2 : v1 < v2;
        const cls1 = sdelp_wins ? 'value-winner' : colorClass(v1, m.key);
        const cls2 = !sdelp_wins ? 'value-winner' : colorClass(v2, m.key);
        return `<tr>
            <td class="model-name">${m.label}</td>
            <td class="${cls1}">${v1.toFixed(m.fmt)}</td>
            <td class="${cls2}">${v2.toFixed(m.fmt)}</td>
        </tr>`;
    }).join('');
}

function colorClass(val, key) {
    if (key === 'MDD (%)' || key === 'AV (%)') {
        return val > 50 ? 'value-negative' : '';
    }
    return val >= 0 ? 'value-positive' : 'value-negative';
}

/* ── Model Info (Mini for right panel) ── */
function renderModelInfoMini(info) {
    const grid = document.getElementById('model-info-mini');
    if (!grid) return;

    const items = [
        { label: 'Lévy α', value: info.levy_alpha },
        { label: 'SDE Steps', value: info.sde_steps },
        { label: 'Window', value: info.window_size },
        { label: 'Actor LR', value: info.actor_lr },
        { label: 'Critic LR', value: info.critic_lr },
        { label: 'γ (Discount)', value: info.gamma },
        { label: 'τ (Soft Update)', value: info.tau },
        { label: 'β (Risk)', value: info.beta },
    ];

    grid.innerHTML = items.map(it => `
        <div class="info-item">
            <div class="info-item-label">${it.label}</div>
            <div class="info-item-value">${it.value}</div>
        </div>
    `).join('');
}

/* ── Model Info (Full) ── */
function renderModelInfo(info) {
    const grid = document.getElementById('model-info-grid');
    if (!grid) return;

    const items = [
        { label: 'Levy Alpha (α)', value: info.levy_alpha },
        { label: 'SDE Steps (K)', value: info.sde_steps },
        { label: 'Risk Penalty (β)', value: info.beta },
        { label: 'Window Size', value: info.window_size },
        { label: 'Txn Cost', value: (info.transaction_cost * 100).toFixed(2) + '%' },
        { label: 'Discount (γ)', value: info.gamma },
        { label: 'Soft Update (τ)', value: info.tau },
        { label: 'Batch Size', value: info.batch_size },
        { label: 'Buffer Size', value: info.buffer_size.toLocaleString() },
        { label: 'Actor LR', value: info.actor_lr },
        { label: 'Critic LR', value: info.critic_lr },
        { label: 'Device', value: info.device.toUpperCase() },
        { label: 'Train Period', value: info.train_period },
        { label: 'Test Period', value: info.test_period },
        { label: 'Num Assets', value: info.num_assets },
    ];

    grid.innerHTML = items.map(it => `
        <div class="info-item">
            <div class="info-item-label">${it.label}</div>
            <div class="info-item-value">${it.value}</div>
        </div>
    `).join('');
}

/* ── IQLBL Model Info (Mini for right panel) ── */
function renderIQLBLModelInfoMini(info) {
    const grid = document.getElementById('model-info-mini');
    if (!grid) return;

    const items = [
        { label: 'Window', value: info.window_size },
        { label: 'IQL γ', value: info.iql_gamma },
        { label: 'IQL τ', value: info.iql_tau },
        { label: 'Expectile', value: info.iql_expectile },
        { label: 'BL τ', value: info.bl_tau },
        { label: 'Risk Aversion', value: info.risk_aversion },
        { label: 'Confidence', value: info.view_confidence },
        { label: 'Short', value: info.allow_short ? 'Yes' : 'No' },
    ];

    grid.innerHTML = items.map(it => `
        <div class="info-item">
            <div class="info-item-label">${it.label}</div>
            <div class="info-item-value">${it.value}</div>
        </div>
    `).join('');
}

/* ── IQLBL Model Info (Full) ── */
function renderIQLBLModelInfo(info) {
    const grid = document.getElementById('model-info-grid');
    if (!grid) return;

    const items = [
        { label: 'Model', value: info.model_name },
        { label: 'Assets', value: info.num_assets + ' (' + info.coins.join(', ') + ')' },
        { label: 'Window Size', value: info.window_size },
        { label: 'IQL Gamma (γ)', value: info.iql_gamma },
        { label: 'IQL Tau (τ)', value: info.iql_tau },
        { label: 'Expectile', value: info.iql_expectile },
        { label: 'Temperature', value: info.iql_temperature },
        { label: 'Learning Rate', value: info.iql_lr },
        { label: 'Epochs', value: info.n_epochs },
        { label: 'Batch Size', value: info.batch_size },
        { label: 'Rebal Window', value: info.learning_window + ' days' },
        { label: 'BL Tau', value: info.bl_tau },
        { label: 'Risk Aversion', value: info.risk_aversion },
        { label: 'View Confidence', value: info.view_confidence },
        { label: 'BL Lookback', value: info.bl_lookback },
        { label: 'Allow Short', value: info.allow_short ? 'Yes' : 'No' },
        { label: 'Fee Rate', value: (info.fee_rate * 100).toFixed(2) + '%' },
        { label: 'Initial Value', value: '$' + info.initial_value.toLocaleString() },
        { label: 'Train Period', value: info.train_period },
        { label: 'Test Period', value: info.test_period },
        { label: 'DD Threshold', value: (info.drawdown_threshold * 100).toFixed(0) + '%' },
        { label: 'Data Source', value: info.data_source },
    ];

    grid.innerHTML = items.map(it => `
        <div class="info-item">
            <div class="info-item-label">${it.label}</div>
            <div class="info-item-value">${it.value}</div>
        </div>
    `).join('');
}

/* ── Refresh ── */
function refreshDashboard() {
    const btn = document.getElementById('refresh-btn');
    if (btn) btn.classList.add('loading');

    fetch('/api/refresh')
        .then(() => loadDashboard())
        .finally(() => {
            if (btn) btn.classList.remove('loading');
        });
}

/* ── Allocation Tab: Large Donut ── */
function renderWeightsDonutAlloc(data) {
    const ctx = document.getElementById('weightsDonutAlloc');
    if (!ctx) return;
    if (charts.donutAlloc) charts.donutAlloc.destroy();

    const labels = data.weight_labels;
    const values = data.current_weights;
    const colors = ['#475569', ...CRYPTO_COLORS.slice(0, labels.length - 1)];

    charts.donutAlloc = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: values.map(v => Math.max(v, 0)),
                backgroundColor: colors,
                borderWidth: 2,
                borderColor: '#0f1422',
                hoverOffset: 8,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '55%',
            plugins: {
                legend: {
                    position: 'right',
                    labels: { padding: 12, font: { size: 11 } },
                },
                tooltip: {
                    callbacks: {
                        label: ctx => `${ctx.label}: ${(ctx.parsed * 100).toFixed(1)}%`,
                    },
                },
            },
        },
    });
}

/* ── Training Period Portfolio Chart ── */
function renderTrainPortfolioChart(data) {
    const ctx = document.getElementById('trainPortfolioChart');
    if (!ctx) return;
    if (charts.trainPortfolio) charts.trainPortfolio.destroy();

    const gradientGreen = ctx.getContext('2d').createLinearGradient(0, 0, 0, 280);
    gradientGreen.addColorStop(0, 'rgba(0, 229, 160, 0.25)');
    gradientGreen.addColorStop(0.5, 'rgba(0, 229, 160, 0.08)');
    gradientGreen.addColorStop(1, 'rgba(0, 229, 160, 0.0)');

    charts.trainPortfolio = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.dates,
            datasets: [
                {
                    label: (MODEL_NAMES[currentModel] || 'Model') + ' (Train)',
                    data: data.sdelp_values,
                    borderColor: COLORS.green,
                    backgroundColor: gradientGreen,
                    borderWidth: 2.5,
                    fill: true,
                    pointRadius: 0,
                    pointHoverRadius: 5,
                    pointHoverBackgroundColor: COLORS.green,
                    pointHoverBorderColor: '#fff',
                    pointHoverBorderWidth: 2,
                    tension: 0.2,
                },
                {
                    label: 'Buy & Hold',
                    data: data.bah_values,
                    borderColor: COLORS.gray,
                    borderWidth: 1.5,
                    borderDash: [6, 4],
                    fill: false,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    tension: 0.2,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: { position: 'top' },
                tooltip: {
                    callbacks: {
                        label: ctx => `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(4)}`,
                    },
                },
            },
            scales: {
                x: {
                    type: 'time',
                    time: { unit: 'month', displayFormats: { month: 'yyyy-MM' } },
                    grid: { display: false },
                    ticks: { maxTicksLimit: 10 },
                    border: { color: COLORS.gridLine },
                },
                y: {
                    grid: { color: COLORS.gridLine },
                    ticks: { padding: 8 },
                    border: { display: false },
                },
            },
        },
    });
}

/* ── Training Metrics Table ── */
function renderTrainMetrics(data) {
    const tbody = document.getElementById('train-metrics-tbody');
    if (!tbody) return;

    const m1 = data.sdelp_metrics;
    const m2 = data.bah_metrics;

    const metricsDef = [
        { keys: ['CRR', 'Cumulative Return'], label: 'CRR (Cumulative Return)', fmt: 4, higherBetter: true, mult: 1 },
        { keys: ['AR (%)', 'Annualized Return (%)', 'Annualized Return'], label: 'Annualized Return (%)', fmt: 2, higherBetter: true, scaleLegacy: true },
        { keys: ['Sharpe', 'Sharpe Ratio'], label: 'Sharpe Ratio', fmt: 3, higherBetter: true, mult: 1 },
        { keys: ['Sortino', 'Sortino Ratio'], label: 'Sortino Ratio', fmt: 3, higherBetter: true, mult: 1 },
        { keys: ['AV (%)', 'Annualized Volatility (%)', 'Annualized Volatility'], label: 'Annualized Volatility (%)', fmt: 2, higherBetter: false, scaleLegacy: true },
        { keys: ['MDD (%)', 'Max Drawdown (%)', 'MDD'], label: 'Max Drawdown (%)', fmt: 2, higherBetter: false, scaleLegacyAbs: true },
    ];

    tbody.innerHTML = metricsDef.map(m => {
        let v1 = getMetric(m1, m.keys);
        let v2 = getMetric(m2, m.keys);

        if (m.scaleLegacy && m1 && m1[m.keys[2]] !== undefined && m1[m.keys[0]] === undefined) v1 *= 100;
        if (m.scaleLegacy && m2 && m2[m.keys[2]] !== undefined && m2[m.keys[0]] === undefined) v2 *= 100;

        if (m.scaleLegacyAbs && m1 && m1[m.keys[2]] !== undefined && m1[m.keys[0]] === undefined) v1 = Math.abs(v1) * 100;
        if (m.scaleLegacyAbs && m2 && m2[m.keys[2]] !== undefined && m2[m.keys[0]] === undefined) v2 = Math.abs(v2) * 100;
        
        // If CRR is 0, attempt to calculate it for train data
        if (m.keys[0] === 'CRR' && v1 === 0 && data.sdelp_values && data.sdelp_values.length > 0) {
            v1 = data.sdelp_values[data.sdelp_values.length - 1] / data.sdelp_values[0];
            v2 = data.bah_values[data.bah_values.length - 1] / data.bah_values[0];
        }

        const sdelp_wins = m.higherBetter ? v1 > v2 : v1 < v2;
        const cls1 = sdelp_wins ? 'value-winner' : colorClass(v1, m.keys[0]);
        const cls2 = !sdelp_wins ? 'value-winner' : colorClass(v2, m.keys[0]);
        return `<tr>
            <td class="model-name">${m.label}</td>
            <td class="${cls1}">${v1.toFixed(m.fmt)}</td>
            <td class="${cls2}">${v2.toFixed(m.fmt)}</td>
        </tr>`;
    }).join('');
}

/* ── Episode Reward Chart ── */
function renderEpisodeRewardChart(data) {
    const ctx = document.getElementById('episodeRewardChart');
    if (!ctx) return;
    if (charts.episodeReward) charts.episodeReward.destroy();

    const episodes = data.episode_rewards.map((_, i) => i + 1);

    charts.episodeReward = new Chart(ctx, {
        type: 'line',
        data: {
            labels: episodes,
            datasets: [{
                label: 'Episode Reward',
                data: data.episode_rewards,
                borderColor: COLORS.green,
                backgroundColor: 'rgba(0, 229, 160, 0.15)',
                borderWidth: 2.5,
                fill: true,
                pointRadius: 5,
                pointBackgroundColor: COLORS.green,
                pointBorderColor: '#fff',
                pointBorderWidth: 2,
                pointHoverRadius: 7,
                tension: 0.3,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        title: ctx => `Episode ${ctx[0].label}`,
                        label: ctx => `Reward: ${ctx.parsed.y.toFixed(2)}`,
                    },
                },
            },
            scales: {
                x: {
                    title: { display: true, text: 'Episode', color: COLORS.textMuted },
                    grid: { display: false },
                    border: { color: COLORS.gridLine },
                },
                y: {
                    title: { display: true, text: 'Cumulative Reward', color: COLORS.textMuted },
                    grid: { color: COLORS.gridLine },
                    border: { display: false },
                },
            },
        },
    });
}

/* ── Episode Portfolio Value Chart ── */
function renderEpisodeValueChart(data) {
    const ctx = document.getElementById('episodeValueChart');
    if (!ctx) return;
    if (charts.episodeValue) charts.episodeValue.destroy();

    const episodes = data.episode_values.map((_, i) => i + 1);

    charts.episodeValue = new Chart(ctx, {
        type: 'line',
        data: {
            labels: episodes,
            datasets: [
                {
                    label: 'Final Portfolio Value',
                    data: data.episode_values,
                    borderColor: COLORS.blue,
                    backgroundColor: 'rgba(79, 142, 255, 0.15)',
                    borderWidth: 2.5,
                    fill: true,
                    pointRadius: 5,
                    pointBackgroundColor: COLORS.blue,
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2,
                    pointHoverRadius: 7,
                    tension: 0.3,
                },
                {
                    label: 'Break-even',
                    data: episodes.map(() => 1.0),
                    borderColor: COLORS.red,
                    borderWidth: 1.5,
                    borderDash: [8, 4],
                    fill: false,
                    pointRadius: 0,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'top' },
                tooltip: {
                    callbacks: {
                        title: ctx => `Episode ${ctx[0].label}`,
                        label: ctx => `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(4)}`,
                    },
                },
            },
            scales: {
                x: {
                    title: { display: true, text: 'Episode', color: COLORS.textMuted },
                    grid: { display: false },
                    border: { color: COLORS.gridLine },
                },
                y: {
                    title: { display: true, text: 'Portfolio Value', color: COLORS.textMuted },
                    grid: { color: COLORS.gridLine },
                    border: { display: false },
                },
            },
        },
    });
}

/* ── Actor Loss Chart ── */
function renderActorLossChart(data) {
    const ctx = document.getElementById('actorLossChart');
    if (!ctx) return;
    if (charts.actorLoss) charts.actorLoss.destroy();

    // Downsample for performance
    const raw = data.actor_losses;
    const step = Math.max(1, Math.floor(raw.length / 500));
    const sampled = raw.filter((_, i) => i % step === 0);
    const labels = sampled.map((_, i) => i * step);

    charts.actorLoss = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Actor Loss',
                data: sampled,
                borderColor: COLORS.purple,
                backgroundColor: 'rgba(167, 139, 250, 0.1)',
                borderWidth: 1.5,
                fill: true,
                pointRadius: 0,
                tension: 0.2,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        title: ctx => `Step ${ctx[0].label}`,
                        label: ctx => `Loss: ${ctx.parsed.y.toFixed(6)}`,
                    },
                },
            },
            scales: {
                x: {
                    title: { display: true, text: 'Update Step', color: COLORS.textMuted },
                    grid: { display: false },
                    ticks: { maxTicksLimit: 8 },
                    border: { color: COLORS.gridLine },
                },
                y: {
                    title: { display: true, text: 'Loss', color: COLORS.textMuted },
                    grid: { color: COLORS.gridLine },
                    border: { display: false },
                },
            },
        },
    });
}

/* ── Critic Loss Chart ── */
function renderCriticLossChart(data) {
    const ctx = document.getElementById('criticLossChart');
    if (!ctx) return;
    if (charts.criticLoss) charts.criticLoss.destroy();

    const raw = data.critic_losses;
    const step = Math.max(1, Math.floor(raw.length / 500));
    const sampled = raw.filter((_, i) => i % step === 0);
    const labels = sampled.map((_, i) => i * step);

    charts.criticLoss = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Critic Loss',
                data: sampled,
                borderColor: COLORS.red,
                backgroundColor: 'rgba(255, 77, 106, 0.1)',
                borderWidth: 1.5,
                fill: true,
                pointRadius: 0,
                tension: 0.2,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        title: ctx => `Step ${ctx[0].label}`,
                        label: ctx => `Loss: ${ctx.parsed.y.toFixed(6)}`,
                    },
                },
            },
            scales: {
                x: {
                    title: { display: true, text: 'Update Step', color: COLORS.textMuted },
                    grid: { display: false },
                    ticks: { maxTicksLimit: 8 },
                    border: { color: COLORS.gridLine },
                },
                y: {
                    title: { display: true, text: 'Loss', color: COLORS.textMuted },
                    grid: { color: COLORS.gridLine },
                    border: { display: false },
                },
            },
        },
    });
}

/* ── Allocation KPIs ── */
function renderAllocKPIs(data) {
    const labels = data.weight_labels;
    const weights = data.current_weights;

    // Top holding
    let maxIdx = 0;
    weights.forEach((w, i) => { if (w > weights[maxIdx]) maxIdx = i; });
    const topEl = document.getElementById('alloc-top-holding');
    const topWEl = document.getElementById('alloc-top-weight');
    if (topEl) topEl.textContent = labels[maxIdx];
    if (topWEl) topWEl.innerHTML = `<span class="text-green">${(weights[maxIdx] * 100).toFixed(1)}%</span>`;

    // Active positions (weight > 1%)
    const active = weights.filter(w => w > 0.01).length;
    const activeEl = document.getElementById('alloc-active-pos');
    const activeDesc = document.getElementById('alloc-active-desc');
    if (activeEl) activeEl.textContent = `${active} / ${labels.length}`;
    if (activeDesc) activeDesc.innerHTML = `<span class="text-blue">Weight > 1%</span>`;

    // HHI (Herfindahl-Hirschman Index)
    const hhi = weights.reduce((sum, w) => sum + w * w, 0);
    const hhiEl = document.getElementById('alloc-hhi');
    const hhiDesc = document.getElementById('alloc-hhi-desc');
    if (hhiEl) hhiEl.textContent = (hhi * 10000).toFixed(0);
    if (hhiDesc) {
        const level = hhi < 0.15 ? 'Diversified' : hhi < 0.25 ? 'Moderate' : 'Concentrated';
        const cls = hhi < 0.15 ? 'text-green' : hhi < 0.25 ? 'text-yellow' : 'text-red';
        hhiDesc.innerHTML = `<span class="${cls}">${level}</span>`;
    }

    // Effective N = 1/HHI
    const effN = 1 / hhi;
    const effEl = document.getElementById('alloc-eff-n');
    const effDesc = document.getElementById('alloc-eff-desc');
    if (effEl) effEl.textContent = effN.toFixed(1);
    if (effDesc) effDesc.innerHTML = `<span class="text-yellow">of ${labels.length} assets</span>`;
}

/* ── Weights Horizontal Bar Chart ── */
function renderWeightsBarChart(data) {
    const ctx = document.getElementById('weightsBarChart');
    if (!ctx) return;
    if (charts.weightsBar) charts.weightsBar.destroy();

    const labels = data.weight_labels;
    const weights = data.current_weights;
    const colors = ['#475569', ...CRYPTO_COLORS.slice(0, labels.length - 1)];

    // Sort by weight descending
    const indices = labels.map((_, i) => i).sort((a, b) => weights[b] - weights[a]);
    const sortedLabels = indices.map(i => labels[i]);
    const sortedWeights = indices.map(i => weights[i] * 100);
    const sortedColors = indices.map(i => colors[i]);

    charts.weightsBar = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: sortedLabels,
            datasets: [{
                label: 'Weight (%)',
                data: sortedWeights,
                backgroundColor: sortedColors.map(c => hexToRgba(c, 0.7)),
                borderColor: sortedColors,
                borderWidth: 1.5,
                borderRadius: 4,
            }],
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: ctx => `${ctx.parsed.x.toFixed(1)}%`,
                    },
                },
            },
            scales: {
                x: {
                    grid: { color: COLORS.gridLine },
                    ticks: { callback: v => v + '%', padding: 4 },
                    border: { display: false },
                },
                y: {
                    grid: { display: false },
                    border: { color: COLORS.gridLine },
                    ticks: { font: { weight: '600' } },
                },
            },
        },
    });
}

/* ── Weight Ranking Table ── */
function renderWeightRanking(data) {
    const tbody = document.getElementById('weight-ranking-tbody');
    if (!tbody) return;

    const labels = data.weight_labels;
    const weights = data.current_weights;
    const colors = ['#475569', ...CRYPTO_COLORS.slice(0, labels.length - 1)];

    const indices = labels.map((_, i) => i).sort((a, b) => weights[b] - weights[a]);
    const maxW = weights[indices[0]];

    tbody.innerHTML = indices.map((idx, rank) => {
        const w = weights[idx];
        const pct = (w * 100).toFixed(1);
        const barWidth = ((w / maxW) * 100).toFixed(1);
        const color = colors[idx];
        return `<tr>
            <td style="color:${COLORS.textMuted}; width:30px;">${rank + 1}</td>
            <td class="model-name">${labels[idx]}</td>
            <td style="font-variant-numeric:tabular-nums; color:${w > 0.1 ? COLORS.green : COLORS.textSecondary}">${pct}%</td>
            <td class="weight-bar-cell" style="width:40%;">
                <div class="weight-bar-bg">
                    <div class="weight-bar-fill" style="width:${barWidth}%; background:${color};"></div>
                </div>
            </td>
        </tr>`;
    }).join('');
}

/* ── Utility ── */
function hexToRgba(hex, alpha) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r},${g},${b},${alpha})`;
}

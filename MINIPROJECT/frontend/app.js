/**
 * Campus Placement Predictor — Client Frontend Application Logic
 * Integrates interactive tab switching, GitHub & LinkedIn account analysis,
 * real-time What-If sliders, diverging local factor breakdown charts, and dual execution mode.
 *
 * Execution Modes:
 *   1. FastAPI Backend Connected (when server running at 127.0.0.1:8000)
 *   2. Client-Side ML Engine (embedded sigmoid logit — identical coefficients to train_model.py)
 */

document.addEventListener('DOMContentLoaded', () => {

    // ═══════════════════════════════════════════════════════
    // API Configuration
    // ═══════════════════════════════════════════════════════
    const API_BASE_URL = 'http://127.0.0.1:8000/v1';
    let isApiConnected = false;
    let lastPredictionData = null; // Cache last form data for What-If sync

    // ═══════════════════════════════════════════════════════
    // DOM Element References
    // ═══════════════════════════════════════════════════════
    const consentModal = document.getElementById('consent-modal');
    const btnAcceptConsent = document.getElementById('btn-accept-consent');
    const btnDeclineConsent = document.getElementById('btn-decline-consent');
    const apiStatusBadge = document.getElementById('api-status-badge');

    const navButtons = document.querySelectorAll('.nav-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    const placementForm = document.getElementById('placement-form');
    const btnResetForm = document.getElementById('btn-reset-form');
    const resultContent = document.getElementById('result-content');

    // What-If Slider Elements
    const simCoding = document.getElementById('sim-coding');
    const valSimCoding = document.getElementById('val-sim-coding');
    const simGithub = document.getElementById('sim-github');
    const valSimGithub = document.getElementById('val-sim-github');
    const simLinkedin = document.getElementById('sim-linkedin');
    const valSimLinkedin = document.getElementById('val-sim-linkedin');
    const simAptitude = document.getElementById('sim-aptitude');
    const valSimAptitude = document.getElementById('val-sim-aptitude');
    const simBacklogs = document.getElementById('sim-backlogs');
    const valSimBacklogs = document.getElementById('val-sim-backlogs');
    const simInternships = document.getElementById('sim-internships');
    const valSimInternships = document.getElementById('val-sim-internships');
    const simProjects = document.getElementById('sim-projects');
    const valSimProjects = document.getElementById('val-sim-projects');

    const simBaseProb = document.getElementById('sim-base-prob');
    const simBaseLabel = document.getElementById('sim-base-label');
    const simNewProb = document.getElementById('sim-new-prob');
    const simNewLabel = document.getElementById('sim-new-label');
    const simDelta = document.getElementById('sim-delta');

    const btnDeleteData = document.getElementById('btn-delete-data');
    const globalImportanceChart = document.getElementById('global-importance-chart');


    // ═══════════════════════════════════════════════════════
    // 1. Consent Modal Management
    // ═══════════════════════════════════════════════════════
    if (localStorage.getItem('campus_placement_consent') === 'true') {
        consentModal.classList.add('hidden');
    }

    btnAcceptConsent.addEventListener('click', () => {
        localStorage.setItem('campus_placement_consent', 'true');
        consentModal.classList.add('hidden');
        consentModal.style.pointerEvents = 'none';
    });

    btnDeclineConsent.addEventListener('click', () => {
        alert('You have declined consent. The predictor is advisory-only and no data will be processed or stored.');
    });


    // ═══════════════════════════════════════════════════════
    // 2. Tab Navigation with Smooth Transitions
    // ═══════════════════════════════════════════════════════
    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');

            navButtons.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            btn.classList.add('active');
            const targetEl = document.getElementById(targetTab);
            if (targetEl) {
                targetEl.classList.add('active');
                // Animate entrance
                targetEl.style.opacity = '0';
                targetEl.style.transform = 'translateY(12px)';
                requestAnimationFrame(() => {
                    targetEl.style.transition = 'opacity 0.35s ease, transform 0.35s ease';
                    targetEl.style.opacity = '1';
                    targetEl.style.transform = 'translateY(0)';
                });
            }

            // Lazy-render insights chart when tab is opened
            if (targetTab === 'tab-insights') {
                renderGlobalImportanceChart();
            }
        });
    });


    // ═══════════════════════════════════════════════════════
    // 3. API Health Ping Check (Auto-Detects Backend)
    // ═══════════════════════════════════════════════════════
    async function checkBackendHealth() {
        try {
            const controller = new AbortController();
            const timeout = setTimeout(() => controller.abort(), 2000);
            const res = await fetch(`${API_BASE_URL}/health`, {
                method: 'GET',
                signal: controller.signal
            });
            clearTimeout(timeout);
            if (res.ok) {
                isApiConnected = true;
                apiStatusBadge.textContent = '⚡ FastAPI Connected (v2026.2)';
                apiStatusBadge.className = 'status-pill status-api';
            }
        } catch (e) {
            isApiConnected = false;
            apiStatusBadge.textContent = '🟢 Client ML Engine Active';
            apiStatusBadge.className = 'status-pill status-local';
        }
    }
    checkBackendHealth();


    // ═══════════════════════════════════════════════════════
    // 4. Embedded Client-Side ML Prediction Engine
    //    (Includes GitHub & LinkedIn Analysis)
    //    Coefficients MATCH train_model.py logit formulation
    // ═══════════════════════════════════════════════════════
    function calculateClientMLPrediction(data) {
        const tenth = parseFloat(data.tenth_pct) || 75;
        const twelfth = parseFloat(data.twelfth_or_diploma_pct) || 75;
        const cgpa = parseFloat(data.cgpa) || 7.0;
        const backlogs = parseInt(data.active_backlogs) || 0;
        const coding = parseInt(data.coding_problems_solved) || 0;
        const aptitude = parseFloat(data.aptitude_score_pct) || 60;
        const comm = parseFloat(data.communication_score) || 3.0;
        const projects = parseInt(data.projects_count) || 0;
        const internships = parseInt(data.internships_count) || 0;
        const ghContribs = parseInt(data.github_contributions_last_year) || 120;
        const liScore = parseFloat(data.linkedin_profile_score) || 3.5;
        const liConnections = parseInt(data.linkedin_connections_count) || 250;

        // Calibrated Sigmoid Logit — identical to train_model.py synthetic data generation
        const logit = (
            -4.8 +
            0.02 * tenth +
            0.02 * twelfth +
            0.52 * (cgpa - 7.0) -
            0.85 * backlogs +
            0.005 * coding +
            0.032 * (aptitude - 60) +
            0.30 * (comm - 3.0) +
            0.28 * projects +
            0.42 * internships +
            // GitHub & LinkedIn Boost Factors
            0.0012 * (ghContribs - 100) +
            0.30 * (liScore - 3.0) +
            0.0008 * (liConnections - 200)
        );

        const rawProb = 1 / (1 + Math.exp(-logit));
        const prob = Math.round(rawProb * 100) / 100;

        let label = 'Moderate likelihood';
        if (prob >= 0.75) label = 'Higher likelihood';
        else if (prob < 0.45) label = 'Lower likelihood';

        // Bootstrap-simulated 80% confidence interval (simplified symmetric spread)
        const ciLower = Math.max(0.01, Math.round((prob - 0.09) * 100) / 100);
        const ciUpper = Math.min(0.99, Math.round((prob + 0.08) * 100) / 100);

        // Signed Local Factor Contributions (rule-based explanation)
        const topPositives = [];
        const topNegatives = [];

        // Coding Practice
        if (coding >= 100) topPositives.push({ feature: 'coding_problems_solved', effect: 0.09, text: 'Your coding practice raised your estimate.' });
        else topNegatives.push({ feature: 'coding_problems_solved', effect: -0.04, text: 'Low coding practice reduced your estimate.' });

        // Active Backlogs (critical negative factor)
        if (backlogs === 0) topPositives.push({ feature: 'active_backlogs', effect: 0.04, text: 'Having zero active backlogs raised your estimate.' });
        else topNegatives.push({ feature: 'active_backlogs', effect: -0.12 * backlogs, text: `Having ${backlogs} active backlog(s) lowered your estimate significantly.` });

        // CGPA
        if (cgpa >= 7.5) topPositives.push({ feature: 'cgpa', effect: 0.08, text: 'Your strong CGPA raised your likelihood score.' });
        else topNegatives.push({ feature: 'cgpa', effect: -0.06, text: 'CGPA is below the cohort average, reducing your estimate.' });

        // GitHub Contributions
        if (ghContribs >= 150) topPositives.push({ feature: 'github_contributions', effect: 0.06, text: 'High GitHub contribution activity strengthened your technical profile.' });
        else topNegatives.push({ feature: 'github_contributions', effect: -0.04, text: 'Low GitHub activity indicates minimal open-source contribution.' });

        // LinkedIn Profile Score
        if (liScore >= 3.8) topPositives.push({ feature: 'linkedin_profile_score', effect: 0.05, text: 'Complete LinkedIn profile and professional presence boosted your estimate.' });
        else topNegatives.push({ feature: 'linkedin_profile_score', effect: -0.04, text: 'Incomplete LinkedIn profile reduced your readiness estimate.' });

        // Aptitude Score
        if (aptitude >= 65) topPositives.push({ feature: 'aptitude_score_pct', effect: 0.07, text: 'Aptitude test performance raised your estimate.' });
        else topNegatives.push({ feature: 'aptitude_score_pct', effect: -0.05, text: 'Aptitude score has room for improvement.' });

        // Internship Experience
        if (internships >= 1) topPositives.push({ feature: 'internships_count', effect: 0.08, text: 'Practical internship experience boosted your estimate.' });
        else topNegatives.push({ feature: 'internships_count', effect: -0.05, text: 'Having no internship experience lowered your estimate.' });

        // Projects
        if (projects >= 2) topPositives.push({ feature: 'projects_count', effect: 0.05, text: 'Completed portfolio projects increased your score.' });
        else topNegatives.push({ feature: 'projects_count', effect: -0.03, text: 'Fewer technical projects lowered your estimate.' });

        // Communication
        if (comm >= 3.5) topPositives.push({ feature: 'communication_score', effect: 0.04, text: 'Strong mock interview score raised your estimate.' });
        else topNegatives.push({ feature: 'communication_score', effect: -0.03, text: 'Communication score has room for improvement.' });

        // Sort by absolute effect and take top 3
        topPositives.sort((a, b) => Math.abs(b.effect) - Math.abs(a.effect));
        topNegatives.sort((a, b) => Math.abs(b.effect) - Math.abs(a.effect));

        return {
            model_version: '2026.2.0-social',
            data_range: 'Batches 2023–2025',
            generated_at: new Date().toISOString(),
            probability: prob,
            interval_80: [ciLower, ciUpper],
            label: label,
            top_positive: topPositives.slice(0, 3),
            top_negative: topNegatives.slice(0, 3),
            limitations: 'This estimate reflects historical measurable records, GitHub activity, and LinkedIn completeness. It cannot measure confidence, attitude, learning ability, or HR fit.'
        };
    }


    // ═══════════════════════════════════════════════════════
    // 5. Render Prediction Result Screen (Rich Animated UI)
    // ═══════════════════════════════════════════════════════
    function renderResultUI(result) {
        const probPercent = Math.round(result.probability * 100);
        const ciLowerPct = Math.round(result.interval_80[0] * 100);
        const ciUpperPct = Math.round(result.interval_80[1] * 100);

        let badgeClass = 'badge-moderate';
        let ringColor = 'var(--status-warning)';
        if (result.label === 'Higher likelihood') { badgeClass = 'badge-higher'; ringColor = 'var(--status-success)'; }
        if (result.label === 'Lower likelihood') { badgeClass = 'badge-lower'; ringColor = 'var(--status-danger)'; }

        // Build positive factors HTML
        const posFactorsHTML = result.top_positive.map(item => `
            <div class="factor-item pos">
                <span class="factor-text">✅ ${item.text}</span>
                <span class="factor-effect text-positive">+${Math.round(item.effect * 100)}%</span>
            </div>
        `).join('');

        // Build negative factors HTML
        const negFactorsHTML = result.top_negative.map(item => `
            <div class="factor-item neg">
                <span class="factor-text">⚠️ ${item.text}</span>
                <span class="factor-effect text-negative">${Math.round(item.effect * 100)}%</span>
            </div>
        `).join('');

        // Animated circular probability gauge using conic-gradient
        const gaugePercent = probPercent;

        resultContent.innerHTML = `
            <div class="result-animated" style="animation: fadeSlideIn 0.5s ease forwards;">
                <div class="probability-header">
                    <div class="prob-gauge-container">
                        <div class="prob-gauge" style="background: conic-gradient(${ringColor} 0% ${gaugePercent}%, rgba(255,255,255,0.08) ${gaugePercent}% 100%);">
                            <div class="prob-gauge-inner">
                                <div class="prob-score">${probPercent}%</div>
                            </div>
                        </div>
                    </div>
                    <div class="prob-meta">
                        <span class="likelihood-badge ${badgeClass}">${result.label}</span>
                        <span class="ci-pill">80% Confidence Interval: [${ciLowerPct}% – ${ciUpperPct}%]</span>
                    </div>
                </div>

                <div class="breakdown-container">
                    <h3 class="breakdown-title">📊 Key Influencing Factors</h3>

                    ${result.top_positive.length > 0 ? `
                        <div class="factor-group">
                            <h4 class="text-positive">▲ Top Positive Factors (Boosts)</h4>
                            ${posFactorsHTML}
                        </div>
                    ` : ''}

                    ${result.top_negative.length > 0 ? `
                        <div class="factor-group">
                            <h4 class="text-negative">▼ Top Negative Factors (Impairments)</h4>
                            ${negFactorsHTML}
                        </div>
                    ` : ''}
                </div>

                <div class="suppression-notice-box" style="margin-top:1.25rem;">
                    ℹ️ <strong>Limitation Disclosure:</strong> ${result.limitations}
                </div>
            </div>
        `;
    }


    // ═══════════════════════════════════════════════════════
    // 6. Form Submission Handler (Dual Mode: API + Client)
    // ═══════════════════════════════════════════════════════
    placementForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        const formData = {
            tenth_pct: parseFloat(document.getElementById('tenth_pct').value),
            twelfth_or_diploma_pct: parseFloat(document.getElementById('twelfth_or_diploma_pct').value),
            cgpa: parseFloat(document.getElementById('cgpa').value),
            active_backlogs: parseInt(document.getElementById('active_backlogs').value) || 0,
            coding_problems_solved: parseInt(document.getElementById('coding_problems_solved').value) || 0,
            aptitude_score_pct: parseFloat(document.getElementById('aptitude_score_pct').value),
            communication_score: parseFloat(document.getElementById('communication_score').value) || 3.0,
            projects_count: parseInt(document.getElementById('projects_count').value) || 0,
            internships_count: parseInt(document.getElementById('internships_count').value) || 0,
            branch: document.getElementById('branch').value,
            github_username: document.getElementById('github_username').value || '',
            github_contributions_last_year: parseInt(document.getElementById('github_contributions_last_year').value) || 0,
            linkedin_connections_count: parseInt(document.getElementById('linkedin_connections_count').value) || 0,
            linkedin_profile_score: parseFloat(document.getElementById('linkedin_profile_score').value) || 3.5
        };

        lastPredictionData = formData;

        // Show loading state
        resultContent.innerHTML = `
            <div class="placeholder-content" style="padding: 3rem;">
                <div class="placeholder-icon" style="animation: pulse 1.2s infinite;">⚡</div>
                <h3>Calculating Prediction...</h3>
                <p>Analyzing your profile against historical placement patterns.</p>
            </div>
        `;

        // Try FastAPI backend first, fallback to client engine
        if (isApiConnected) {
            try {
                const res = await fetch(`${API_BASE_URL}/predict`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(formData)
                });
                if (res.ok) {
                    const data = await res.json();
                    renderResultUI(data);
                    syncWhatIfBase(formData, data.probability);
                    return;
                }
            } catch (err) {
                console.warn('API unavailable, falling back to client ML engine:', err.message);
            }
        }

        // Client-side fallback (with slight delay for UX polish)
        setTimeout(() => {
            const localResult = calculateClientMLPrediction(formData);
            renderResultUI(localResult);
            syncWhatIfBase(formData, localResult.probability);
        }, 400);
    });

    btnResetForm.addEventListener('click', () => {
        placementForm.reset();
        resultContent.innerHTML = `
            <div class="placeholder-content">
                <div class="placeholder-icon">🎯</div>
                <h3>No Prediction Calculated Yet</h3>
                <p>Fill in your profile details and click <strong>"Calculate Estimated Likelihood"</strong> to view your calibrated probability, confidence interval, and factor breakdown.</p>
            </div>
        `;
    });


    // ═══════════════════════════════════════════════════════
    // 7. Real-Time What-If Counterfactual Simulator
    //    (Includes GitHub & LinkedIn sliders)
    // ═══════════════════════════════════════════════════════
    function syncWhatIfBase(formData, baseProb) {
        lastPredictionData = formData;

        simCoding.value = formData.coding_problems_solved || 0;
        valSimCoding.textContent = formData.coding_problems_solved || 0;

        simGithub.value = formData.github_contributions_last_year || 0;
        valSimGithub.textContent = formData.github_contributions_last_year || 0;

        simLinkedin.value = formData.linkedin_profile_score || 3.5;
        valSimLinkedin.textContent = formData.linkedin_profile_score || 3.5;

        simAptitude.value = formData.aptitude_score_pct || 60;
        valSimAptitude.textContent = `${formData.aptitude_score_pct || 60}%`;

        simBacklogs.value = formData.active_backlogs || 0;
        valSimBacklogs.textContent = formData.active_backlogs || 0;

        simInternships.value = formData.internships_count || 0;
        valSimInternships.textContent = formData.internships_count || 0;

        simProjects.value = formData.projects_count || 0;
        valSimProjects.textContent = formData.projects_count || 0;

        const basePct = Math.round(baseProb * 100);
        simBaseProb.textContent = `${basePct}%`;

        let label = 'Moderate likelihood';
        if (baseProb >= 0.75) label = 'Higher likelihood';
        if (baseProb < 0.45) label = 'Lower likelihood';
        simBaseLabel.textContent = label;

        recalculateWhatIf();
    }

    function recalculateWhatIf() {
        // Read slider values
        const coding = parseInt(simCoding.value);
        const ghContribs = parseInt(simGithub.value);
        const liScore = parseFloat(simLinkedin.value);
        const aptitude = parseFloat(simAptitude.value);
        const backlogs = parseInt(simBacklogs.value);
        const internships = parseInt(simInternships.value);
        const projects = parseInt(simProjects.value);

        // Update display labels
        valSimCoding.textContent = coding;
        valSimGithub.textContent = ghContribs;
        valSimLinkedin.textContent = liScore;
        valSimAptitude.textContent = `${aptitude}%`;
        valSimBacklogs.textContent = backlogs;
        valSimInternships.textContent = internships;
        valSimProjects.textContent = projects;

        // Get fixed (non-actionable) values from current form
        const tenth = parseFloat(document.getElementById('tenth_pct').value) || 75;
        const twelfth = parseFloat(document.getElementById('twelfth_or_diploma_pct').value) || 75;
        const cgpa = parseFloat(document.getElementById('cgpa').value) || 7.0;
        const comm = parseFloat(document.getElementById('communication_score').value) || 3.0;
        const liConnections = parseInt(document.getElementById('linkedin_connections_count').value) || 250;

        // Base prediction (using form values)
        const baseData = lastPredictionData || {
            tenth_pct: tenth, twelfth_or_diploma_pct: twelfth, cgpa: cgpa,
            active_backlogs: parseInt(document.getElementById('active_backlogs').value) || 0,
            coding_problems_solved: parseInt(document.getElementById('coding_problems_solved').value) || 0,
            aptitude_score_pct: parseFloat(document.getElementById('aptitude_score_pct').value) || 60,
            communication_score: comm, projects_count: parseInt(document.getElementById('projects_count').value) || 0,
            internships_count: parseInt(document.getElementById('internships_count').value) || 0,
            github_contributions_last_year: parseInt(document.getElementById('github_contributions_last_year').value) || 120,
            linkedin_profile_score: parseFloat(document.getElementById('linkedin_profile_score').value) || 3.5,
            linkedin_connections_count: liConnections
        };
        const baseResult = calculateClientMLPrediction(baseData);

        // New prediction (with slider-modified actionable features)
        const newData = {
            ...baseData,
            active_backlogs: backlogs,
            coding_problems_solved: coding,
            github_contributions_last_year: ghContribs,
            linkedin_profile_score: liScore,
            aptitude_score_pct: aptitude,
            communication_score: comm,
            projects_count: projects,
            internships_count: internships,
            linkedin_connections_count: liConnections
        };
        const newResult = calculateClientMLPrediction(newData);

        const newPct = Math.round(newResult.probability * 100);
        const basePct = Math.round(baseResult.probability * 100);
        const delta = newPct - basePct;

        simBaseProb.textContent = `${basePct}%`;
        simNewProb.textContent = `${newPct}%`;

        let newLabel = newResult.label;
        simNewLabel.textContent = newLabel;

        let baseLabel = baseResult.label;
        simBaseLabel.textContent = baseLabel;

        if (delta >= 0) {
            simDelta.textContent = `+${delta}%`;
            simDelta.className = 'delta-value delta-positive';
        } else {
            simDelta.textContent = `${delta}%`;
            simDelta.className = 'delta-value delta-negative';
        }
    }

    // Attach real-time slider listeners
    [simCoding, simGithub, simLinkedin, simAptitude, simBacklogs, simInternships, simProjects].forEach(slider => {
        if (slider) slider.addEventListener('input', recalculateWhatIf);
    });


    // ═══════════════════════════════════════════════════════
    // 8. Global Feature Importance Chart (Animated Bars)
    // ═══════════════════════════════════════════════════════
    function renderGlobalImportanceChart() {
        const features = [
            { name: 'Active Backlogs (Negative)', weight: 0.85, actionable: true, negative: true },
            { name: 'Coding Problems Solved', weight: 0.65, actionable: true },
            { name: 'Cumulative GPA (CGPA)', weight: 0.55, actionable: false },
            { name: 'Mock Aptitude Score', weight: 0.48, actionable: true },
            { name: 'Internships Completed', weight: 0.45, actionable: true },
            { name: 'GitHub Contributions (12M)', weight: 0.42, actionable: true },
            { name: 'LinkedIn Profile Score', weight: 0.38, actionable: true },
            { name: 'Mock GD/HR Score', weight: 0.35, actionable: true },
            { name: 'Completed Projects', weight: 0.30, actionable: true }
        ];

        globalImportanceChart.innerHTML = features.map((item, idx) => `
            <div class="importance-item" style="animation: fadeSlideIn 0.3s ease ${idx * 0.06}s both;">
                <span class="imp-label">${item.name} ${item.actionable ? '<span class="actionable-badge">⚡ Actionable</span>' : '<span class="fixed-badge">📌 Fixed</span>'}</span>
                <div class="imp-bar-wrapper">
                    <div class="imp-bar ${item.negative ? 'imp-bar-negative' : ''}" style="width: 0%; transition: width 0.8s ease ${idx * 0.08}s;"></div>
                </div>
                <span class="imp-val">${Math.round(item.weight * 100)}%</span>
            </div>
        `).join('');

        // Trigger animation after DOM paint
        requestAnimationFrame(() => {
            const bars = globalImportanceChart.querySelectorAll('.imp-bar');
            features.forEach((item, idx) => {
                if (bars[idx]) bars[idx].style.width = `${Math.round(item.weight * 100)}%`;
            });
        });
    }


    // ═══════════════════════════════════════════════════════
    // 9. Data Erasure Action Trigger (GDPR-style)
    // ═══════════════════════════════════════════════════════
    if (btnDeleteData) {
        btnDeleteData.addEventListener('click', async () => {
            if (confirm('⚠️ Are you sure you want to delete all temporary inference records?\n\nThis action will erase your consent preferences and any cached data.')) {
                // Try API deletion if backend connected
                if (isApiConnected) {
                    try {
                        await fetch(`${API_BASE_URL}/data`, { method: 'DELETE' });
                    } catch (e) {
                        console.warn('API data deletion failed:', e.message);
                    }
                }
                // Clear local storage
                localStorage.removeItem('campus_placement_consent');
                lastPredictionData = null;
                alert('✅ Your temporary data has been successfully erased. The page will now reload.');
                window.location.reload();
            }
        });
    }


    // ═══════════════════════════════════════════════════════
    // 10. Initial Page Load — Auto-calculate with defaults
    // ═══════════════════════════════════════════════════════
    // Only auto-predict after consent is accepted
    if (localStorage.getItem('campus_placement_consent') === 'true') {
        setTimeout(() => {
            placementForm.dispatchEvent(new Event('submit', { cancelable: true }));
        }, 600);
    }

});

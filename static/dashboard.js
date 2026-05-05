/* ═══════════════════════════════════════════════════
   AI Tutor — Dashboard Logic
   ═══════════════════════════════════════════════════ */

const API = '/api';

// ─── Auth State ──────────────────────────────────
const currentUserId = localStorage.getItem('ai_tutor_user_id');

function authHeaders() {
    const h = { 'Content-Type': 'application/json' };
    if (currentUserId) h['X-User-Id'] = currentUserId;
    return h;
}

document.addEventListener('DOMContentLoaded', () => {
    if (!currentUserId) {
        window.location.href = '/'; // redirect to login
        return;
    }
    loadDashboard();
    loadRoadmap();
    loadInsights();
    loadDueCount();
});

// ─── Load Dashboard Data ─────────────────────────
async function loadDashboard() {
    try {
        const res = await fetch(`${API}/dashboard`, { headers: authHeaders() });
        const data = await res.json();

        // Stats
        document.getElementById('stat-overall').textContent =
            (data.overall_mastery * 100).toFixed(0) + '%';
        const attempted = data.skills.filter(s => s.total_attempts > 0).length;
        document.getElementById('stat-skills').textContent = `${attempted}/20`;
        document.getElementById('stat-interactions').textContent =
            data.recent_interactions.length;

        // Render radar chart
        renderRadarChart(data.skills);

        // Render recent activity
        renderRecentActivity(data.recent_interactions);
    } catch (e) {
        console.error('Dashboard load error:', e);
    }
}


// ─── Radar Chart ─────────────────────────────────
function renderRadarChart(skills) {
    if (!skills || skills.length === 0) {
        document.getElementById('chart-radar').innerHTML =
            '<p class="empty-text">Chưa có dữ liệu. Hãy bắt đầu học!</p>';
        return;
    }

    // Group by chapter
    const chapterMastery = {};
    for (const s of skills) {
        // Look up chapter from skill_id — we'll use the skill_name as label
        const chapter = s.skill_name;
        chapterMastery[chapter] = s.p_mastery;
    }

    const labels = Object.keys(chapterMastery);
    const values = Object.values(chapterMastery).map(v => +(v * 100).toFixed(0));

    const trace = {
        type: 'scatterpolar',
        r: [...values, values[0]], // close the polygon
        theta: [...labels, labels[0]],
        fill: 'toself',
        fillcolor: 'rgba(99, 102, 241, 0.15)',
        line: { color: '#6366f1', width: 2 },
        marker: { size: 6, color: '#6366f1' },
    };

    const layout = {
        polar: {
            radialaxis: { visible: true, range: [0, 100], ticksuffix: '%', gridcolor: '#e5e7eb' },
            angularaxis: { gridcolor: '#e5e7eb' },
            bgcolor: 'transparent',
        },
        showlegend: false,
        margin: { t: 30, b: 30, l: 60, r: 60 },
        paper_bgcolor: 'transparent',
        font: { family: 'Inter, sans-serif', size: 11 },
        height: 360,
    };

    Plotly.newPlot('chart-radar', [trace], layout, { responsive: true, displayModeBar: false });
}


// ─── Skill Tree / Roadmap ────────────────────────
async function loadRoadmap() {
    try {
        const res = await fetch(`${API}/roadmap`, { headers: authHeaders() });
        const data = await res.json();
        renderSkillTree(data.nodes, data.chapters);
    } catch (e) {
        document.getElementById('skill-tree').innerHTML =
            '<p class="empty-text">Không thể tải cây kỹ năng.</p>';
    }
}

function renderSkillTree(nodes, chapters) {
    const container = document.getElementById('skill-tree');
    container.innerHTML = '';

    // Group by chapter
    const grouped = {};
    for (const node of nodes) {
        if (!grouped[node.chapter]) grouped[node.chapter] = [];
        grouped[node.chapter].push(node);
    }

    for (const chapter of Object.keys(grouped)) {
        const chapterDiv = document.createElement('div');
        chapterDiv.className = 'skill-tree-chapter';

        const title = document.createElement('h4');
        title.className = 'skill-tree-chapter-title';
        title.textContent = chapter;
        chapterDiv.appendChild(title);

        const nodesDiv = document.createElement('div');
        nodesDiv.className = 'skill-tree-nodes';

        const skills = grouped[chapter];
        for (let i = 0; i < skills.length; i++) {
            const skill = skills[i];
            const node = document.createElement('div');
            node.className = `skill-tree-node level-${skill.level}`;
            node.innerHTML = `
                <div class="skill-node-name">${skill.skill_name}</div>
                <div class="skill-node-bar">
                    <div class="skill-node-bar-fill" style="width:${(skill.p_mastery * 100).toFixed(0)}%"></div>
                </div>
                <div class="skill-node-pct">${(skill.p_mastery * 100).toFixed(0)}%</div>
            `;
            nodesDiv.appendChild(node);

            // Add arrow connector if not last
            if (i < skills.length - 1) {
                const arrow = document.createElement('div');
                arrow.className = 'skill-tree-arrow';
                arrow.innerHTML = '→';
                nodesDiv.appendChild(arrow);
            }
        }

        chapterDiv.appendChild(nodesDiv);
        container.appendChild(chapterDiv);
    }
}


// ─── Insights ────────────────────────────────────
async function loadInsights() {
    try {
        const res = await fetch(`${API}/insights`, { headers: authHeaders() });
        const data = await res.json();
        const container = document.getElementById('insights-list');

        if (!data.insights || data.insights.length === 0) {
            container.innerHTML = '<p class="empty-text">Chưa có insights. Hãy luyện tập thêm!</p>';
            return;
        }

        container.innerHTML = data.insights.map(insight => `
            <div class="insight-card type-${insight.insight_type}">
                <div class="insight-message">${formatMarkdown(insight.message)}</div>
            </div>
        `).join('');
    } catch (e) {
        document.getElementById('insights-list').innerHTML =
            '<p class="empty-text">Không thể tải insights.</p>';
    }
}


// ─── Due Count ───────────────────────────────────
async function loadDueCount() {
    try {
        const res = await fetch(`${API}/review/due`, { headers: authHeaders() });
        const data = await res.json();
        document.getElementById('stat-due').textContent = data.due_count;
    } catch (e) { /* silent */ }
}


// ─── Recent Activity ─────────────────────────────
function renderRecentActivity(interactions) {
    const container = document.getElementById('recent-activity');
    if (!interactions || interactions.length === 0) {
        container.innerHTML = '<p class="empty-text">Chưa có hoạt động.</p>';
        return;
    }

    container.innerHTML = interactions.slice(0, 10).map(i => {
        const modeEmoji = {
            socratic: '🎓', exam: '📝', assess: '✅',
            plan: '📋', quiz: '📝', review: '📖',
            answer: '💬', visualize: '📊', diagnostic: '🔍',
        }[i.response_mode] || '💬';

        const time = i.timestamp ? new Date(i.timestamp).toLocaleString('vi-VN') : '';
        return `
            <div class="activity-item">
                <span class="activity-emoji">${modeEmoji}</span>
                <div class="activity-content">
                    <div class="activity-question">${escHtml(i.question)}</div>
                    <div class="activity-meta">${time}</div>
                </div>
                ${i.is_correct !== null ? `<span class="activity-result ${i.is_correct ? 'correct' : 'wrong'}">${i.is_correct ? '✅' : '❌'}</span>` : ''}
            </div>
        `;
    }).join('');
}


// ─── Utils ───────────────────────────────────────
function formatMarkdown(text) {
    return text
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/\n/g, '<br>');
}

function escHtml(s) {
    if (!s) return '';
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

/* ═══════════════════════════════════════════════════
   AI Tutor — Chatbot + Quiz + Visualization Logic
   ═══════════════════════════════════════════════════ */

const API = '/api';

// ─── Auth State ──────────────────────────────────
let currentUserId = localStorage.getItem('ai_tutor_user_id');
let currentUsername = localStorage.getItem('ai_tutor_username');

function authHeaders() {
    const h = { 'Content-Type': 'application/json' };
    if (currentUserId) h['X-User-Id'] = currentUserId;
    return h;
}

// ─── State ───────────────────────────────────────
let pendingImages = [];   // array of File objects (multi-image)
let isLoading = false;
let currentSessionId = null; // Conversation memory session

// Quiz state
let quizState = {
    sessionId: null,
    questions: [],
    currentIndex: 0,
    correctCount: 0,
    sessionType: 'practice', // practice | diagnostic
    selectedSkillId: null,
    selectedDifficulty: 0,  // 0 = adaptive (auto)
};

document.addEventListener('DOMContentLoaded', () => {
    setupTextarea();
    setupPasteHandler();
    setupDragAndDrop();
    checkLoginState();
});


// ═══════════════════════════════════════════════════
//  LOGIN / LOGOUT
// ═══════════════════════════════════════════════════

function checkLoginState() {
    if (currentUserId && currentUsername) {
        // Already logged in
        hideLoginOverlay();
        showUserBadge();
        checkDueReviews();
        loadSessions();
    } else {
        showLoginOverlay();
    }
}

function showLoginOverlay() {
    document.getElementById('login-overlay').style.display = 'flex';
}

function hideLoginOverlay() {
    document.getElementById('login-overlay').style.display = 'none';
}

function showUserBadge() {
    const badge = document.getElementById('user-badge');
    const name = document.getElementById('user-badge-name');
    if (badge && name) {
        name.textContent = `👤 ${currentUsername}`;
        badge.style.display = 'flex';
    }
    // Sidebar footer
    const footer = document.getElementById('sidebar-footer');
    const sidebarName = document.getElementById('sidebar-user-name');
    if (footer && sidebarName) {
        sidebarName.textContent = currentUsername;
        footer.style.display = 'flex';
    }
}

async function doLogin() {
    const input = document.getElementById('login-username');
    const errorEl = document.getElementById('login-error');
    const btn = document.getElementById('login-btn');
    const username = input.value.trim();

    if (!username || username.length < 2) {
        errorEl.textContent = 'Tên đăng nhập cần ít nhất 2 ký tự';
        errorEl.style.display = 'block';
        return;
    }

    btn.disabled = true;
    btn.textContent = '⏳ Đang đăng nhập...';
    errorEl.style.display = 'none';

    try {
        const res = await fetch(`${API}/auth/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username }),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || 'Lỗi đăng nhập');
        }
        const data = await res.json();

        // Save to localStorage
        currentUserId = String(data.user_id);
        currentUsername = data.display_name || data.username;
        localStorage.setItem('ai_tutor_user_id', currentUserId);
        localStorage.setItem('ai_tutor_username', currentUsername);

        hideLoginOverlay();
        showUserBadge();
        checkDueReviews();
        loadSessions();

        // Welcome toast for new users
        if (data.is_new) {
            setTimeout(() => {
                addBubble('assistant', `🎉 Chào mừng **${currentUsername}**! Mình là AI Tutor, gia sư Toán 12 của em. Hãy bắt đầu bằng cách hỏi một câu hỏi, hoặc làm bài **Test Chẩn Đoán** để mình đánh giá năng lực nhé!`);
            }, 300);
        }
    } catch (e) {
        errorEl.textContent = e.message;
        errorEl.style.display = 'block';
    }
    btn.disabled = false;
    btn.textContent = 'Bắt đầu học 🚀';
}

function doLogout() {
    localStorage.removeItem('ai_tutor_user_id');
    localStorage.removeItem('ai_tutor_username');
    currentUserId = null;
    currentUsername = null;
    currentSessionId = null;
    document.getElementById('user-badge').style.display = 'none';
    document.getElementById('chat-messages').innerHTML = '';
    // Clear sidebar
    const footer = document.getElementById('sidebar-footer');
    if (footer) footer.style.display = 'none';
    const list = document.getElementById('session-list');
    if (list) list.innerHTML = '';
    const welcome = document.getElementById('welcome-msg');
    if (welcome) welcome.style.display = 'flex';
    showLoginOverlay();
}


// ═══════════════════════════════════════════════════
//  TEXTAREA & INPUT
// ═══════════════════════════════════════════════════

function setupTextarea() {
    const ta = document.getElementById('chat-input');
    ta.addEventListener('input', () => {
        ta.style.height = 'auto';
        ta.style.height = Math.min(ta.scrollHeight, 120) + 'px';
        updateSendButton();
    });
}

function updateSendButton() {
    const ta = document.getElementById('chat-input');
    const hasText = ta.value.trim().length > 0;
    const hasImages = pendingImages.length > 0;
    document.getElementById('btn-send').disabled = !(hasText || hasImages);
}

function handleInputKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
}


// ═══════════════════════════════════════════════════
//  PASTE & DRAG-DROP
// ═══════════════════════════════════════════════════

function setupPasteHandler() {
    document.addEventListener('paste', (e) => {
        const items = e.clipboardData?.items;
        if (!items) return;
        let found = false;
        for (const item of items) {
            if (item.type.startsWith('image/')) {
                e.preventDefault();
                const file = item.getAsFile();
                if (file) { attachImage(file); found = true; }
            }
        }
    });
}

function setupDragAndDrop() {
    const wrapper = document.querySelector('.input-wrapper');
    wrapper.addEventListener('dragover', (e) => { e.preventDefault(); wrapper.classList.add('drag-over'); });
    wrapper.addEventListener('dragleave', () => { wrapper.classList.remove('drag-over'); });
    wrapper.addEventListener('drop', (e) => {
        e.preventDefault();
        wrapper.classList.remove('drag-over');
        const files = e.dataTransfer?.files;
        if (files) {
            for (const file of files) {
                if (file.type.startsWith('image/')) attachImage(file);
            }
        }
    });
}

function handleFileSelect(e) {
    const files = e.target.files;
    if (files) {
        for (const file of files) {
            if (file.type.startsWith('image/')) attachImage(file);
        }
    }
    e.target.value = '';
}


// ═══════════════════════════════════════════════════
//  IMAGE ATTACH & PREVIEW
// ═══════════════════════════════════════════════════

function attachImage(file) {
    // Giới hạn tối đa 5 ảnh
    if (pendingImages.length >= 5) {
        alert('Tối đa 5 ảnh mỗi lần gửi.');
        return;
    }
    pendingImages.push(file);
    renderImagePreviews();
    document.getElementById('chat-input').focus();
}

function renderImagePreviews() {
    const container = document.getElementById('image-previews');
    container.innerHTML = '';
    if (pendingImages.length === 0) {
        container.style.display = 'none';
        updateSendButton();
        return;
    }
    pendingImages.forEach((file, idx) => {
        const wrapper = document.createElement('div');
        wrapper.className = 'preview-item';
        const img = document.createElement('img');
        img.className = 'preview-thumb';
        img.alt = 'Ảnh đính kèm';
        img.src = URL.createObjectURL(file);
        const removeBtn = document.createElement('button');
        removeBtn.className = 'preview-remove';
        removeBtn.innerHTML = '✕';
        removeBtn.title = 'Xoá ảnh';
        removeBtn.onclick = () => removeImage(idx);
        const label = document.createElement('span');
        label.className = 'preview-label';
        label.textContent = file.name.length > 18 ? file.name.slice(0, 16) + '…' : file.name;
        const badge = document.createElement('span');
        badge.className = 'preview-badge';
        badge.textContent = `${idx + 1}/${pendingImages.length}`;
        wrapper.appendChild(badge);
        wrapper.appendChild(img);
        wrapper.appendChild(removeBtn);
        wrapper.appendChild(label);
        container.appendChild(wrapper);
    });
    container.style.display = 'flex';
    updateSendButton();
}

function removeImage(idx) {
    if (idx === undefined) {
        // Xóa tất cả (backward compat)
        pendingImages = [];
    } else {
        pendingImages.splice(idx, 1);
    }
    renderImagePreviews();
}


// ═══════════════════════════════════════════════════
//  CHAT
// ═══════════════════════════════════════════════════

function newConversation() {
    currentSessionId = null;
    const container = document.getElementById('chat-messages');
    container.innerHTML = '';
    const welcome = document.getElementById('welcome-msg');
    if (welcome) welcome.style.display = 'flex';

    // Update active state in sidebar
    document.querySelectorAll('.session-item.active').forEach(el => el.classList.remove('active'));
}

async function loadSessions() {
    if (!currentUserId) return;
    try {
        const res = await fetch(`${API}/sessions`, { headers: authHeaders() });
        if (!res.ok) return;
        const data = await res.json();
        renderSessions(data.sessions || []);
    } catch (e) {
        console.error('Failed to load sessions', e);
    }
}

function renderSessions(sessions) {
    const list = document.getElementById('session-list');
    if (!list) return;
    list.innerHTML = '';

    if (sessions.length === 0) {
        list.innerHTML = '<div style="font-size: 12px; color: var(--text-muted); text-align: center; padding: 10px;">Chưa có chat nào</div>';
        return;
    }

    sessions.forEach(s => {
        const item = document.createElement('div');
        item.className = `session-item ${s.id === currentSessionId ? 'active' : ''}`;
        item.dataset.id = s.id;  // store id for active tracking
        item.onclick = () => loadSessionHistory(s.id);

        let dateStr = '';
        if (s.last_active) {
            const d = new Date(s.last_active);
            const today = new Date();
            const isToday = d.toDateString() === today.toDateString();
            dateStr = isToday
                ? d.toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' })
                : d.toLocaleDateString('vi-VN');
        }

        item.innerHTML = `
            <div style="flex:1; overflow:hidden; display:flex; flex-direction:column; gap:2px;">
                <div class="session-title">${esc(s.title || 'Chat mới')}</div>
                <div class="session-meta">${dateStr}</div>
            </div>
            <button class="delete-session-btn" onclick="deleteSession(event, ${s.id})" title="Xoá chat">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <polyline points="3 6 5 6 21 6"></polyline>
                    <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                    <line x1="10" y1="11" x2="10" y2="17"></line>
                    <line x1="14" y1="11" x2="14" y2="17"></line>
                </svg>
            </button>
        `;
        list.appendChild(item);
    });
}

async function loadSessionHistory(sessionId) {
    if (isLoading) return;
    currentSessionId = sessionId;

    // Cập nhật active state trực tiếp trên DOM không cần re-render
    document.querySelectorAll('.session-item').forEach(el => {
        el.classList.toggle('active', parseInt(el.dataset.id) === sessionId);
    });

    const container = document.getElementById('chat-messages');
    container.innerHTML = '<div style="text-align:center; padding: 20px; color: var(--text-muted);">&#8203;</div>';

    const welcome = document.getElementById('welcome-msg');
    if (welcome) welcome.style.display = 'none';

    try {
        const res = await fetch(`${API}/session/${sessionId}/history`, { headers: authHeaders() });
        if (!res.ok) throw new Error('Không thể tải lịch sử');
        const data = await res.json();

        container.innerHTML = '';
        if (data.messages && data.messages.length > 0) {
            data.messages.forEach(m => {
                addBubble(m.role, m.content, m.skill_id);
                if (m.visualization) renderVisualization(m.visualization);
            });
            container.scrollTop = container.scrollHeight;
        } else {
            container.innerHTML = '<div style="text-align:center; padding: 40px; color: var(--text-muted); font-size: 14px;">Phên chat này chưa có tin nhắn</div>';
        }
    } catch (e) {
        container.innerHTML = `<div style="color:var(--red); text-align:center; padding:20px;">⚠️ Lỗi: ${e.message}</div>`;
    }
}

async function deleteSession(event, sessionId) {
    event.stopPropagation();
    if (!confirm('Bạn có chắc chắn muốn xóa phiên chat này không?')) return;

    try {
        const res = await fetch(`${API}/session/${sessionId}`, {
            method: 'DELETE',
            headers: authHeaders()
        });
        if (!res.ok) throw new Error('Không thể xóa phiên chat');

        if (currentSessionId === sessionId) {
            newConversation();
        }

        loadSessions();
    } catch (e) {
        alert(`Lỗi: ${e.message}`);
    }
}

async function clearAllSessions() {
    if (!confirm('Bạn có chắc chắn muốn xóa TẤT CẢ lịch sử chat không? Hành động này không thể hoàn tác.')) return;

    try {
        const res = await fetch(`${API}/sessions/all`, {
            method: 'DELETE',
            headers: authHeaders()
        });
        if (!res.ok) throw new Error('Không thể xóa lịch sử chat');

        newConversation();
        loadSessions();
    } catch (e) {
        alert(`Lỗi: ${e.message}`);
    }
}

function sendSuggestion(text) {
    document.getElementById('chat-input').value = text;
    updateSendButton();
    sendMessage();
}

async function sendMessage() {
    if (isLoading) return;
    const input = document.getElementById('chat-input');
    const msg = input.value.trim();
    if (!msg && pendingImages.length === 0) return;

    const welcome = document.getElementById('welcome-msg');
    if (welcome) welcome.style.display = 'none';

    const capturedText = msg;
    const capturedImages = [...pendingImages];
    input.value = '';
    input.style.height = 'auto';
    removeImage();
    document.getElementById('btn-send').disabled = true;
    isLoading = true;

    addBubble('user', capturedText, null, capturedImages);

    try {
        if (capturedImages.length > 0) {
            // ── Multi-Image Hybrid Vision (không stream) ───────────
            const typing = addTyping();
            const formData = new FormData();
            capturedImages.forEach(f => formData.append('files', f));
            if (capturedText) formData.append('user_text', capturedText);
            if (currentSessionId) formData.append('session_id', currentSessionId);

            const uploadHeaders = {};
            if (currentUserId) uploadHeaders['X-User-Id'] = currentUserId;

            const res = await fetch(`${API}/upload`, {
                method: 'POST',
                body: formData,
                headers: uploadHeaders,
            });
            if (!res.ok) {
                const err = await res.json().catch(() => ({}));
                throw new Error(err.detail || `Lỗi server (${res.status})`);
            }
            const data = await res.json();
            typing.remove();
            if (data.session_id) currentSessionId = data.session_id;
            if (data.ocr_text) addOcrInfo(data.ocr_text);
            addBubble('assistant', data.response, data.skill_name);
            if (data.visualization) renderVisualization(data.visualization);
        } else {
            // ── Text-only: Streaming SSE ───────────────────────────
            await sendMessageStream(capturedText);
        }
    } catch (err) {
        addBubble('assistant', `⚠️ ${err.message}`);
    }

    loadSessions();
    isLoading = false;
}


async function sendMessageStream(capturedText) {
    const container = document.getElementById('chat-messages');
    const typingEl = addTyping();

    try {
        const res = await fetch(`${API}/chat/stream`, {
            method: 'POST',
            headers: authHeaders(),
            body: JSON.stringify({
                message: capturedText,
                mode: 'auto',
                session_id: currentSessionId,
            }),
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            typingEl.remove();
            throw new Error(err.detail || `Lỗi server (${res.status})`);
        }

        // Giữ typing indicator đến khi token đầu tiên về (TTFT của model
        // reasoning có thể 10-20s — bubble trống với con trỏ ▋ trông như treo).
        let bodyEl = null;
        const ensureStreamBubble = () => {
            if (bodyEl) return bodyEl;
            typingEl.remove();
            const msgDiv = document.createElement('div');
            msgDiv.className = 'message assistant';
            const bodyId = 'stream-body-' + Date.now();
            msgDiv.innerHTML = `
                <div class="msg-avatar">🤖</div>
                <div class="msg-body" id="${bodyId}"><span class="stream-cursor">▋</span></div>
            `;
            container.appendChild(msgDiv);
            container.scrollTop = container.scrollHeight;
            bodyEl = document.getElementById(bodyId);
            return bodyEl;
        };
        let fullText = '';

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop();

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                const jsonStr = line.slice(6).trim();
                if (!jsonStr) continue;

                try {
                    const frame = JSON.parse(jsonStr);
                    if (frame.type === 'meta') {
                        const sid = res.headers.get('X-Session-Id') || frame.session_id;
                        if (sid) currentSessionId = parseInt(sid) || currentSessionId;
                        // Đã phân loại xong, gia sư bắt đầu soạn bài — báo trạng thái
                        const label = typingEl.querySelector('.typing-status');
                        if (!bodyEl && !label) {
                            const status = document.createElement('span');
                            status.className = 'typing-status';
                            status.textContent = '💭 Thầy đang suy nghĩ...';
                            const body = typingEl.querySelector('.msg-body');
                            if (body) body.appendChild(status);
                        }
                    } else if (frame.type === 'token') {
                        fullText += frame.content;
                        ensureStreamBubble().innerHTML = formatText(fullText) + '<span class="stream-cursor">▋</span>';
                        container.scrollTop = container.scrollHeight;
                    } else if (frame.type === 'done') {
                        fullText = frame.full_response || fullText;
                        const el = ensureStreamBubble();
                        el.innerHTML = formatText(fullText);
                        renderKatex(el);
                        if (frame.visualization) {
                            renderVisualization(frame.visualization);
                        }
                        if (frame.mode_used === 'quiz') {
                            autoStartQuizFromChat(frame.skill_id);
                        }
                        container.scrollTop = container.scrollHeight;
                    } else if (frame.type === 'error') {
                        ensureStreamBubble().innerHTML = `<span style="color:var(--red)">⚠️ ${esc(frame.message)}</span>`;
                    }
                } catch (_) { }
            }
        }

        // Stream kết thúc mà không có token/done nào → dọn typing indicator
        if (!bodyEl) typingEl.remove();

        // Lấy session_id từ header
        const sid = res.headers.get('X-Session-Id');
        if (sid) currentSessionId = parseInt(sid) || currentSessionId;

    } catch (err) {
        typingEl.remove();
        throw err;
    }
}


// Quiz-in-chat: intent "quiz" từ chat tự mở bài quiz với độ khó adaptive
function autoStartQuizFromChat(skillId) {
    quizState.selectedSkillId = skillId || 'derivative_basic';
    quizState.selectedChapter = null;
    quizState.selectedDifficulty = 0; // 0 = adaptive theo BKT
    generateQuiz();
}




// ═══════════════════════════════════════════════════
//  RENDER HELPERS
// ═══════════════════════════════════════════════════

function addBubble(role, content, skillName, imageFiles) {
    const container = document.getElementById('chat-messages');
    const div = document.createElement('div');
    div.className = `message ${role}`;
    const avatar = role === 'user' ? '👤' : '🤖';

    // imageFiles có thể là array hoặc single File (backward compat)
    const files = imageFiles
        ? (Array.isArray(imageFiles) ? imageFiles : [imageFiles])
        : [];

    let imageHTML = '';
    if (files.length > 0) {
        const imgs = files.map(f =>
            `<img class="msg-image" src="${URL.createObjectURL(f)}" alt="Ảnh đã gửi">`
        ).join('');
        imageHTML = `<div class="msg-images-grid">${imgs}</div>`;
    }

    const formatted = content ? formatText(content) : '';
    let meta = '';
    if (role === 'assistant' && skillName) {
        meta = `<div class="msg-meta"><span class="msg-skill-badge">📐 ${esc(skillName)}</span></div>`;
    }

    div.innerHTML = `
        <div class="msg-avatar">${avatar}</div>
        <div>
            ${imageHTML ? `<div class="msg-body msg-body-image">${imageHTML}</div>` : ''}
            ${formatted ? `<div class="msg-body">${formatted}</div>` : ''}
            ${meta}
        </div>
    `;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
    if (role === 'assistant') {
        const bodyEl = div.querySelector('.msg-body');
        if (bodyEl) renderKatex(bodyEl);
    }
}

function addOcrInfo(text) {
    const container = document.getElementById('chat-messages');
    const div = document.createElement('div');
    div.className = 'ocr-info';
    div.innerHTML = `<span class="ocr-label">🔍 Nội dung nhận dạng (OCR):</span> ${esc(text)}`;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
}

function addTyping() {
    const container = document.getElementById('chat-messages');
    const div = document.createElement('div');
    div.className = 'message assistant';
    div.innerHTML = `
        <div class="msg-avatar">🤖</div>
        <div class="msg-body typing-indicator">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        </div>
    `;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
    return div;
}


// ═══════════════════════════════════════════════════
//  PLOTLY VISUALIZATION
// ═══════════════════════════════════════════════════

function renderVisualization(visData) {
    if (!visData || typeof visData !== 'object') return false;
    if (visData.vis_type === 'plot') {
        renderPlotlyChart(visData);
        return true;
    }

    renderVisualizationError('Định dạng trực quan hóa chưa được hỗ trợ.');
    return false;
}


function renderVisualizationError(message) {
    const container = document.getElementById('chat-messages');
    const wrapper = document.createElement('div');
    wrapper.className = 'message assistant';
    wrapper.innerHTML = `
        <div class="msg-avatar">📊</div>
        <div class="msg-body chart-error">⚠️ ${esc(message)}</div>
    `;
    container.appendChild(wrapper);
    container.scrollTop = container.scrollHeight;
}


function renderPlotlyChart(visData) {
    const container = document.getElementById('chat-messages');

    if (!visData.data || !Array.isArray(visData.data.traces)) {
        renderVisualizationError('Dữ liệu đồ thị từ máy chủ không hợp lệ.');
        return;
    }

    const wrapper = document.createElement('div');
    wrapper.className = 'message assistant';

    const chartDiv = document.createElement('div');
    chartDiv.className = 'chart-container';
    chartDiv.id = 'chart-' + Date.now();

    const caption = visData.latex_caption
        ? `<div class="chart-caption">$${visData.latex_caption}$</div>`
        : '';

    wrapper.innerHTML = `
        <div class="msg-avatar">📊</div>
        <div class="msg-body chart-bubble">
            ${caption}
        </div>
    `;
    wrapper.querySelector('.msg-body').prepend(chartDiv);
    container.appendChild(wrapper);
    container.scrollTop = container.scrollHeight;

    // Never leave a silent blank bubble when the CDN or payload fails.
    if (typeof Plotly === 'undefined') {
        chartDiv.classList.add('chart-error');
        chartDiv.textContent = '⚠️ Không tải được thư viện hiển thị đồ thị. Vui lòng tải lại trang.';
    } else {
        const layout = {
            ...visData.data.layout,
            autosize: true,
            height: 360,
        };
        try {
            const plotPromise = Plotly.newPlot(chartDiv, visData.data.traces, layout, {
                responsive: true,
                displayModeBar: true,
                modeBarButtonsToRemove: ['lasso2d', 'select2d'],
            });
            Promise.resolve(plotPromise).catch(() => {
                chartDiv.classList.add('chart-error');
                chartDiv.textContent = '⚠️ Không thể dựng đồ thị từ dữ liệu đã nhận.';
            });
        } catch (_) {
            chartDiv.classList.add('chart-error');
            chartDiv.textContent = '⚠️ Không thể dựng đồ thị từ dữ liệu đã nhận.';
        }
    }

    // Render KaTeX in caption
    const captionEl = wrapper.querySelector('.chart-caption');
    if (captionEl) renderKatex(captionEl);
}


// ═══════════════════════════════════════════════════
//  QUIZ SYSTEM
// ═══════════════════════════════════════════════════

async function showQuizPanel() {
    const modal = document.getElementById('skill-modal');
    const grid = document.getElementById('skill-grid');
    grid.innerHTML = '<p style="color:#888">Đang tải...</p>';
    modal.style.display = 'flex';

    try {
        const res = await fetch(`${API}/skills`, { headers: authHeaders() });
        const data = await res.json();
        grid.innerHTML = '';

        // Group by chapter - We only show chapters now, not specific skills
        const chapters = data.chapters || [];
        
        const section = document.createElement('div');
        section.className = 'skill-chapter';
        
        for (const chapter of chapters) {
            const btn = document.createElement('button');
            btn.className = 'skill-btn';
            btn.textContent = chapter;
            btn.dataset.chapter = chapter;
            btn.onclick = () => selectChapter(chapter, btn);
            section.appendChild(btn);
        }
        grid.appendChild(section);
    } catch (e) {
        grid.innerHTML = '<p style="color:#f87171">Không thể tải danh sách chương.</p>';
    }
}

function selectChapter(chapter, btn) {
    document.querySelectorAll('.skill-btn.selected').forEach(b => b.classList.remove('selected'));
    btn.classList.add('selected');
    quizState.selectedChapter = chapter;
    quizState.selectedSkillId = null; // Clear out skill_id
    document.getElementById('modal-submit-btn').disabled = false;
}

function selectDifficulty(diff) {
    quizState.selectedDifficulty = diff;
    document.querySelectorAll('.diff-btn').forEach(b => {
        b.classList.toggle('active', parseInt(b.dataset.diff) === diff);
    });
    // Show/hide adaptive hint
    const hint = document.getElementById('adaptive-hint');
    if (hint) hint.style.display = diff === 0 ? 'block' : 'none';
}

function closeSkillModal() {
    document.getElementById('skill-modal').style.display = 'none';
}

async function generateQuiz() {
    closeSkillModal();
    const panel = document.getElementById('quiz-panel');
    const qArea = document.getElementById('quiz-question-area');
    const rArea = document.getElementById('quiz-result-area');

    panel.style.display = 'block';
    rArea.style.display = 'none';
    qArea.innerHTML = '<p class="quiz-loading">⏳ Đang sinh đề bài...</p>';

    try {
        const payload = {
            difficulty: quizState.selectedDifficulty,
            count: 5,
        };
        if (quizState.selectedChapter) {
            payload.chapter = quizState.selectedChapter;
        } else if (quizState.selectedSkillId) {
            payload.skill_id = quizState.selectedSkillId;
        }

        const res = await fetch(`${API}/quiz/generate`, {
            method: 'POST',
            headers: authHeaders(),
            body: JSON.stringify(payload),
        });
        if (!res.ok) throw new Error('Không thể sinh đề.');
        const data = await res.json();

        quizState.sessionId = data.session_id;
        quizState.questions = data.questions;
        quizState.currentIndex = 0;
        quizState.correctCount = 0;
        quizState.totalScore = 0;
        quizState.maxScore = data.max_score || 10;
        quizState.sessionType = 'practice';

        document.getElementById('quiz-title').textContent = data.adaptive
                ? `📝 Bài Kiểm Tra (🎯 Adaptive: ${data.adaptive.difficulty_label})`
                : '📝 Bài Kiểm Tra';
        renderQuizQuestion(0);
    } catch (e) {
        qArea.innerHTML = `<p class="quiz-error">⚠️ ${e.message}</p>`;
    }
}

function renderQuizQuestion(index) {
    const q = quizState.questions[index];
    if (!q) return;
    const total = quizState.questions.length;
    const qArea = document.getElementById('quiz-question-area');

    const pct = ((index) / total) * 100;
    document.getElementById('quiz-progress-fill').style.width = pct + '%';
    document.getElementById('quiz-score-text').textContent = `Câu ${index + 1}/${total}`;
    document.getElementById('quiz-correct-text').textContent = quizState.examFormat
        ? `${quizState.totalScore || 0}đ`
        : `✅ ${quizState.correctCount} đúng`;

    const qType = q.question_type || 'mcq';
    const typeBadge = { mcq: 'Phần 1 — Trắc nghiệm', true_false: 'Phần 2 — Đúng/Sai', short_answer: 'Phần 3 — Trả lời ngắn' };
    const pointsLabel = q.points ? ` (${q.points}đ)` : '';

    let bodyHTML = '';
    if (qType === 'mcq') {
        bodyHTML = renderMCQBody(q, index);
    } else if (qType === 'true_false') {
        bodyHTML = renderTFBody(q, index);
    } else {
        bodyHTML = renderSABody(q, index);
    }

    qArea.innerHTML = `
        <div class="quiz-question-card">
            <div class="quiz-q-number">
                <span class="quiz-type-badge type-${qType}">${typeBadge[qType] || 'Câu hỏi'}${pointsLabel}</span>
                Câu ${index + 1}/${total}
            </div>
            <div class="quiz-q-text" id="quiz-q-text-${index}">${q.question_latex}</div>
            ${bodyHTML}
            <button class="quiz-submit-btn" id="quiz-submit-${index}" onclick="submitQuizAnswer(${index})" disabled>
                Xác nhận
            </button>
            <div class="quiz-feedback" id="quiz-feedback-${index}" style="display:none"></div>
        </div>
    `;

    renderKatex(document.getElementById(`quiz-q-text-${index}`));
    if (qType === 'mcq' && q.choices) {
        q.choices.forEach((_, i) => {
            const el = document.getElementById(`quiz-opt-${index}-${i}`);
            if (el) renderKatex(el);
        });
    }
    if (qType === 'true_false' && q.statements) {
        q.statements.forEach((_, i) => {
            const el = document.getElementById(`quiz-stmt-${index}-${i}`);
            if (el) renderKatex(el);
        });
    }
}

function renderMCQBody(q, index) {
    return `
        <div class="quiz-options" id="quiz-options-${index}">
            ${q.choices.map((c, i) => `
                <button class="quiz-option-btn" data-index="${i}" onclick="selectQuizOption(${index}, ${i}, this)">
                    <span class="quiz-option-label">${['A', 'B', 'C', 'D'][i]}</span>
                    <span class="quiz-option-text" id="quiz-opt-${index}-${i}">${c}</span>
                </button>
            `).join('')}
        </div>
    `;
}

function renderTFBody(q, index) {
    return `
        <div class="tf-statements" id="quiz-tf-${index}">
            ${q.statements.map((s, i) => `
                <div class="tf-statement-row" id="tf-row-${index}-${i}">
                    <div class="tf-statement-text" id="quiz-stmt-${index}-${i}">
                        <span class="tf-label">${['a)', 'b)', 'c)', 'd)'][i]}</span>
                        ${s.text}
                    </div>
                    <div class="tf-toggle-group">
                        <button class="tf-toggle-btn" data-val="true" onclick="selectTF(${index}, ${i}, true, this)">
                            Đúng
                        </button>
                        <button class="tf-toggle-btn" data-val="false" onclick="selectTF(${index}, ${i}, false, this)">
                            Sai
                        </button>
                    </div>
                </div>
            `).join('')}
        </div>
    `;
}

function renderSABody(q, index) {
    return `
        <div class="sa-input-area" id="quiz-sa-${index}">
            <input type="text" class="sa-input" id="sa-input-${index}"
                   placeholder="Nhập đáp án (số)..."
                   oninput="onSAInput(${index})">
        </div>
    `;
}


// ── Selection handlers ──
let selectedOptionIndex = -1;
let tfSelections = {};  // { `${qIndex}-${stmtIndex}`: true/false }

function selectQuizOption(qIndex, optIndex, btn) {
    const optionsDiv = document.getElementById(`quiz-options-${qIndex}`);
    optionsDiv.querySelectorAll('.quiz-option-btn').forEach(b => b.classList.remove('selected'));
    btn.classList.add('selected');
    selectedOptionIndex = optIndex;
    document.getElementById(`quiz-submit-${qIndex}`).disabled = false;
}

function selectTF(qIndex, stmtIndex, value, btn) {
    const row = document.getElementById(`tf-row-${qIndex}-${stmtIndex}`);
    row.querySelectorAll('.tf-toggle-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    tfSelections[`${qIndex}-${stmtIndex}`] = value;

    // Enable submit when ALL 4 statements answered
    const q = quizState.questions[qIndex];
    const answered = q.statements.every((_, i) => `${qIndex}-${i}` in tfSelections);
    document.getElementById(`quiz-submit-${qIndex}`).disabled = !answered;
}

function onSAInput(qIndex) {
    const val = document.getElementById(`sa-input-${qIndex}`).value.trim();
    document.getElementById(`quiz-submit-${qIndex}`).disabled = val.length === 0;
}


// ── Submit Answer (all types) ──
async function submitQuizAnswer(qIndex) {
    const q = quizState.questions[qIndex];
    const qType = q.question_type || 'mcq';
    const submitBtn = document.getElementById(`quiz-submit-${qIndex}`);
    submitBtn.disabled = true;
    submitBtn.textContent = '⏳...';

    // Build request body
    const body = { session_id: quizState.sessionId, question_id: q.id };
    if (qType === 'mcq') {
        if (selectedOptionIndex < 0) return;
        body.selected_index = selectedOptionIndex;
    } else if (qType === 'true_false') {
        body.tf_answers = q.statements.map((_, i) => tfSelections[`${qIndex}-${i}`] || false);
    } else {
        body.text_answer = document.getElementById(`sa-input-${qIndex}`).value.trim();
    }

    try {
        const res = await fetch(`${API}/quiz/submit`, {
            method: 'POST',
            headers: authHeaders(),
            body: JSON.stringify(body),
        });
        const data = await res.json();

        // Type-specific feedback
        if (qType === 'mcq') {
            const optionsDiv = document.getElementById(`quiz-options-${qIndex}`);
            optionsDiv.querySelectorAll('.quiz-option-btn').forEach((btn, i) => {
                btn.disabled = true;
                btn.onclick = null;
                if (i === data.correct_index) btn.classList.add('correct');
                if (i === selectedOptionIndex && !data.is_correct) btn.classList.add('wrong');
            });
        } else if (qType === 'true_false') {
            const correct = data.correct_statements || [];
            q.statements.forEach((_, i) => {
                const row = document.getElementById(`tf-row-${qIndex}-${i}`);
                const studentVal = tfSelections[`${qIndex}-${i}`];
                const isRight = studentVal === correct[i];
                row.classList.add(isRight ? 'tf-correct' : 'tf-wrong');
                row.querySelectorAll('.tf-toggle-btn').forEach(b => b.disabled = true);
                // Show correct answer
                const correctLabel = document.createElement('span');
                correctLabel.className = 'tf-correct-label';
                correctLabel.textContent = correct[i] ? '→ Đúng' : '→ Sai';
                row.appendChild(correctLabel);
            });
        } else {
            const saInput = document.getElementById(`sa-input-${qIndex}`);
            saInput.disabled = true;
            saInput.classList.add(data.is_correct ? 'sa-correct' : 'sa-wrong');
            if (!data.is_correct && data.correct_answer) {
                const correctDiv = document.createElement('div');
                correctDiv.className = 'sa-correct-answer';
                correctDiv.textContent = `Đáp án đúng: ${data.correct_answer}`;
                saInput.parentElement.appendChild(correctDiv);
            }
        }

        if (data.is_correct) quizState.correctCount++;
        quizState.totalScore = data.session_progress.score_so_far;

        // Points feedback
        const fb = document.getElementById(`quiz-feedback-${qIndex}`);
        fb.style.display = 'block';
        const ptsText = data.points_earned > 0
            ? ` (+${data.points_earned}đ / ${data.max_points}đ)`
            : ` (0đ / ${data.max_points}đ)`;
        fb.className = `quiz-feedback ${data.points_earned > 0 ? 'correct' : 'wrong'}`;
        fb.innerHTML = `
            <strong>${data.points_earned > 0 ? '✅' : '❌'} ${ptsText}</strong>
            ${data.explanation ? `<div class="quiz-explanation">${data.explanation}</div>` : ''}
            ${data.new_mastery !== null && data.new_mastery !== undefined ? `<div class="quiz-mastery">Mastery: ${(data.new_mastery * 100).toFixed(0)}%</div>` : ''}
        `;
        renderKatex(fb);

        // Update score bar
        document.getElementById('quiz-correct-text').textContent = quizState.examFormat
            ? `${quizState.totalScore}đ`
            : `✅ ${quizState.correctCount} đúng`;

        // Next button
        submitBtn.textContent = (qIndex + 1 < quizState.questions.length) ? 'Câu tiếp →' : 'Xem kết quả';
        submitBtn.disabled = false;
        submitBtn.onclick = () => {
            selectedOptionIndex = -1;
            tfSelections = {};
            if (qIndex + 1 < quizState.questions.length) {
                quizState.currentIndex = qIndex + 1;
                renderQuizQuestion(qIndex + 1);
            } else {
                showQuizResult();
            }
        };

    } catch (e) {
        submitBtn.textContent = '⚠️ Lỗi, thử lại';
        submitBtn.disabled = false;
    }
}

async function showQuizResult() {
    const qArea = document.getElementById('quiz-question-area');
    const rArea = document.getElementById('quiz-result-area');
    qArea.style.display = 'none';
    rArea.style.display = 'block';

    document.getElementById('quiz-progress-fill').style.width = '100%';

    try {
        const res = await fetch(`${API}/quiz/result/${quizState.sessionId}`, { headers: authHeaders() });
        const data = await res.json();

        const scorePct = data.score_percent;
        const emoji = scorePct >= 80 ? '🎉' : scorePct >= 60 ? '👍' : '💪';

        const partScoresHTML = data.part_scores ? `
            <div class="part-scores">
                <div class="part-score-item">
                    <span class="part-label">Phần 1 (Trắc nghiệm):</span>
                    <span class="part-value">${data.part_scores.mcq || 0}đ</span>
                </div>
                <div class="part-score-item">
                    <span class="part-label">Phần 2 (Đúng/Sai):</span>
                    <span class="part-value">${data.part_scores.true_false || 0}đ</span>
                </div>
                <div class="part-score-item">
                    <span class="part-label">Phần 3 (Trả lời ngắn):</span>
                    <span class="part-value">${data.part_scores.short_answer || 0}đ</span>
                </div>
            </div>
        ` : '';

        rArea.innerHTML = `
            <div class="quiz-result-card">
                <div class="quiz-result-emoji">${emoji}</div>
                <div class="quiz-result-score">${data.total_score}/${data.max_score}đ</div>
                <div class="quiz-result-summary">${scorePct.toFixed(0)}% — ${data.total_questions} câu hỏi</div>
                ${partScoresHTML}
                <div class="quiz-skill-results">
                    ${data.skill_results.map(s => `
                        <div class="quiz-skill-row">
                            <span class="quiz-skill-name">${s.skill_name}</span>
                            <span class="quiz-skill-score">${s.correct}/${s.total}</span>
                            <div class="quiz-skill-bar">
                                <div class="quiz-skill-bar-fill" style="width:${(s.mastery * 100).toFixed(0)}%"></div>
                            </div>
                            <span class="quiz-skill-pct">${(s.mastery * 100).toFixed(0)}%</span>
                        </div>
                    `).join('')}
                </div>
                <div class="quiz-result-actions">
                    <button class="quiz-action-btn" onclick="closeQuizPanel()">Đóng</button>
                    <button class="quiz-action-btn primary" onclick="showQuizPanel()">Làm bài khác</button>
                </div>
            </div>
        `;
    } catch (e) {
        rArea.innerHTML = '<p class="quiz-error">Không thể tải kết quả.</p>';
    }
}

function closeQuizPanel() {
    document.getElementById('quiz-panel').style.display = 'none';
    document.getElementById('quiz-question-area').style.display = 'block';
    document.getElementById('quiz-result-area').style.display = 'none';
}


// ═══════════════════════════════════════════════════
//  DIAGNOSTIC TEST
// ═══════════════════════════════════════════════════

async function startDiagnostic() {
    const welcome = document.getElementById('welcome-msg');
    if (welcome) welcome.style.display = 'none';

    const panel = document.getElementById('quiz-panel');
    const qArea = document.getElementById('quiz-question-area');
    const rArea = document.getElementById('quiz-result-area');

    panel.style.display = 'block';
    rArea.style.display = 'none';
    qArea.innerHTML = '<p class="quiz-loading">🔍 Đang tạo bài test chẩn đoán...</p>';
    document.getElementById('quiz-title').textContent = '🔍 Test Chẩn Đoán Năng Lực';

    try {
        const res = await fetch(`${API}/diagnostic/start`, { method: 'POST', headers: authHeaders() });
        const data = await res.json();

        quizState.sessionId = data.session_id;
        quizState.questions = [data.first_question];
        quizState.currentIndex = 0;
        quizState.correctCount = 0;
        quizState.sessionType = 'diagnostic';

        document.getElementById('quiz-score-text').textContent = `Câu 1/${data.total_questions}`;
        renderDiagnosticQuestion(data.first_question, 1, data.total_questions);
    } catch (e) {
        qArea.innerHTML = `<p class="quiz-error">⚠️ ${e.message}</p>`;
    }
}

function renderDiagnosticQuestion(q, current, total) {
    const qArea = document.getElementById('quiz-question-area');
    const pct = ((current - 1) / total) * 100;
    document.getElementById('quiz-progress-fill').style.width = pct + '%';
    document.getElementById('quiz-score-text').textContent = `Câu ${current}/${total}`;
    document.getElementById('quiz-correct-text').textContent = `✅ ${quizState.correctCount} đúng`;

    qArea.innerHTML = `
        <div class="quiz-question-card">
            <div class="quiz-q-number">Câu ${current}/${total}</div>
            <div class="quiz-q-text" id="diag-q-text">${q.question_latex}</div>
            <div class="quiz-options" id="diag-options">
                ${q.choices.map((c, i) => `
                    <button class="quiz-option-btn" data-index="${i}" onclick="selectDiagOption(${i}, this)">
                        <span class="quiz-option-label">${['A', 'B', 'C', 'D'][i]}</span>
                        <span class="quiz-option-text" id="diag-opt-${i}">${c}</span>
                    </button>
                `).join('')}
            </div>
            <button class="quiz-submit-btn" id="diag-submit" onclick="submitDiagAnswer(${q.id}, ${current}, ${total})" disabled>
                Xác nhận
            </button>
            <div class="quiz-feedback" id="diag-feedback" style="display:none"></div>
        </div>
    `;

    renderKatex(document.getElementById('diag-q-text'));
    q.choices.forEach((_, i) => {
        const el = document.getElementById(`diag-opt-${i}`);
        if (el) renderKatex(el);
    });
}

let diagSelectedIndex = -1;

function selectDiagOption(idx, btn) {
    document.querySelectorAll('#diag-options .quiz-option-btn').forEach(b => b.classList.remove('selected'));
    btn.classList.add('selected');
    diagSelectedIndex = idx;
    document.getElementById('diag-submit').disabled = false;
}

async function submitDiagAnswer(questionId, current, total) {
    if (diagSelectedIndex < 0) return;
    const submitBtn = document.getElementById('diag-submit');
    submitBtn.disabled = true;
    submitBtn.textContent = '⏳...';

    try {
        const res = await fetch(`${API}/diagnostic/answer`, {
            method: 'POST',
            headers: authHeaders(),
            body: JSON.stringify({
                session_id: quizState.sessionId,
                question_id: questionId,
                selected_index: diagSelectedIndex,
            }),
        });
        const data = await res.json();

        // Highlight answers
        const optionsDiv = document.getElementById('diag-options');
        optionsDiv.querySelectorAll('.quiz-option-btn').forEach((btn, i) => {
            btn.disabled = true;
            btn.onclick = null;
            if (i === data.correct_index) btn.classList.add('correct');
            if (i === diagSelectedIndex && !data.is_correct) btn.classList.add('wrong');
        });

        if (data.is_correct) quizState.correctCount++;
        document.getElementById('quiz-correct-text').textContent = `✅ ${quizState.correctCount} đúng`;

        const fb = document.getElementById('diag-feedback');
        fb.style.display = 'block';
        fb.className = `quiz-feedback ${data.is_correct ? 'correct' : 'wrong'}`;
        fb.innerHTML = `<strong>${data.is_correct ? '✅ Đúng!' : '❌ Sai!'}</strong>
            ${data.explanation ? `<div class="quiz-explanation">${data.explanation}</div>` : ''}`;
        renderKatex(fb);

        if (data.is_completed) {
            // All questions answered — show results button only
            submitBtn.textContent = 'Xem kết quả chẩn đoán';
            submitBtn.disabled = false;
            submitBtn.onclick = () => showDiagnosticResult();
        } else if (data.next_question && current < total) {
            // Still have questions remaining
            submitBtn.textContent = 'Câu tiếp →';
            submitBtn.disabled = false;
            submitBtn.onclick = () => {
                diagSelectedIndex = -1;
                renderDiagnosticQuestion(data.next_question, current + 1, total);
            };
        } else {
            // Fallback: session should be done if no next_question
            submitBtn.textContent = 'Xem kết quả chẩn đoán';
            submitBtn.disabled = false;
            submitBtn.onclick = () => showDiagnosticResult();
        }
    } catch (e) {
        submitBtn.textContent = '⚠️ Lỗi';
    }
}

async function showDiagnosticResult() {
    const qArea = document.getElementById('quiz-question-area');
    const rArea = document.getElementById('quiz-result-area');
    qArea.style.display = 'none';
    rArea.style.display = 'block';
    document.getElementById('quiz-progress-fill').style.width = '100%';

    try {
        const res = await fetch(`${API}/diagnostic/result/${quizState.sessionId}`, { headers: authHeaders() });
        const data = await res.json();

        const scorePct = data.overall_score;
        const emoji = scorePct >= 80 ? '🎉' : scorePct >= 50 ? '📊' : '📝';

        rArea.innerHTML = `
            <div class="quiz-result-card diagnostic-result">
                <div class="quiz-result-emoji">${emoji}</div>
                <h3 style="margin:8px 0">Kết Quả Chẩn Đoán</h3>
                <div class="quiz-result-score">${scorePct.toFixed(0)}%</div>

                ${data.weak_areas.length > 0 ? `
                    <div class="diagnostic-weak">
                        <h4 style="color:#f87171">⚠️ Cần ôn lại:</h4>
                        <ul>
                            ${data.weak_areas.map(w => `<li><strong>${w.skill_name}</strong> — ${w.recommendation}</li>`).join('')}
                        </ul>
                    </div>
                ` : '<p style="color:#34d399">✅ Tuyệt vời! Không có lỗ hổng kiến thức lớn.</p>'}

                <div class="diagnostic-profile">
                    <h4>Hồ sơ năng lực:</h4>
                    ${data.skill_profile.map(s => `
                        <div class="quiz-skill-row">
                            <span class="quiz-skill-name">${s.skill_name}</span>
                            <span class="${s.is_correct ? 'diag-correct' : 'diag-wrong'}">${s.is_correct ? '✅' : '❌'}</span>
                            <div class="quiz-skill-bar">
                                <div class="quiz-skill-bar-fill ${s.level}" style="width:${(s.estimated_mastery * 100).toFixed(0)}%"></div>
                            </div>
                            <span class="quiz-skill-pct">${(s.estimated_mastery * 100).toFixed(0)}%</span>
                        </div>
                    `).join('')}
                </div>

                <div class="quiz-result-actions">
                    <button class="quiz-action-btn" onclick="closeQuizPanel()">Đóng</button>
                    <a class="quiz-action-btn primary" href="/dashboard">📊 Xem Dashboard</a>
                </div>
            </div>
        `;
    } catch (e) {
        rArea.innerHTML = '<p class="quiz-error">Không thể tải kết quả.</p>';
    }
}


// ═══════════════════════════════════════════════════
//  SPACED REPETITION REVIEW
// ═══════════════════════════════════════════════════

async function checkDueReviews() {
    try {
        const res = await fetch(`${API}/review/due`, { headers: authHeaders() });
        const data = await res.json();
        if (data.due_count > 0) {
            showReviewBanner(data.due_count);
        }
    } catch (e) { /* silent */ }
}

function showReviewBanner(count) {
    const container = document.getElementById('chat-messages');
    const existingBanner = document.getElementById('review-banner');
    if (existingBanner) return;

    const banner = document.createElement('div');
    banner.className = 'review-banner';
    banner.id = 'review-banner';
    banner.innerHTML = `
        <div class="review-banner-content">
            <span class="review-banner-icon">📖</span>
            <div>
                <strong>${count} kiến thức cần ôn tập!</strong>
                <p>Hệ thống phát hiện có kiến thức em sắp quên. Ôn ngay để nhớ lâu hơn!</p>
            </div>
            <button class="review-banner-btn" onclick="startReview()">Ôn tập ngay</button>
            <button class="review-banner-dismiss" onclick="this.parentElement.parentElement.remove()">✕</button>
        </div>
    `;
    container.insertBefore(banner, container.firstChild);
}

async function startReview() {
    const welcome = document.getElementById('welcome-msg');
    if (welcome) welcome.style.display = 'none';

    const panel = document.getElementById('quiz-panel');
    const qArea = document.getElementById('quiz-question-area');
    const rArea = document.getElementById('quiz-result-area');

    panel.style.display = 'block';
    rArea.style.display = 'none';
    qArea.innerHTML = '<p class="quiz-loading">📖 Đang tải thẻ ôn tập...</p>';
    document.getElementById('quiz-title').textContent = '📖 Ôn Tập Chống Quên';

    try {
        const res = await fetch(`${API}/review/due`, { headers: authHeaders() });
        const data = await res.json();

        if (data.due_count === 0) {
            qArea.innerHTML = `
                <div class="review-empty">
                    <div class="review-empty-icon">🎉</div>
                    <h3>Không có gì cần ôn!</h3>
                    <p>Em đã ôn tập đầy đủ. Hẹn lại sau nhé!</p>
                    <button class="quiz-action-btn" onclick="closeQuizPanel()">Đóng</button>
                </div>
            `;
            return;
        }

        quizState.sessionType = 'review';
        renderReviewCard(data.cards, 0);
    } catch (e) {
        qArea.innerHTML = `<p class="quiz-error">⚠️ ${e.message}</p>`;
    }
}

function renderReviewCard(cards, index) {
    if (index >= cards.length) {
        const qArea = document.getElementById('quiz-question-area');
        qArea.innerHTML = `
            <div class="review-empty">
                <div class="review-empty-icon">✅</div>
                <h3>Hoàn thành ôn tập!</h3>
                <p>Tuyệt vời, em đã ôn xong tất cả các thẻ hôm nay.</p>
                <button class="quiz-action-btn" onclick="closeQuizPanel()">Đóng</button>
            </div>
        `;
        return;
    }

    const card = cards[index];
    const q = card.question;
    const qArea = document.getElementById('quiz-question-area');

    const pct = (index / cards.length) * 100;
    document.getElementById('quiz-progress-fill').style.width = pct + '%';
    document.getElementById('quiz-score-text').textContent = `Thẻ ${index + 1}/${cards.length}`;

    qArea.innerHTML = `
        <div class="quiz-question-card review-card-wrapper">
            <div class="review-skill-tag">${card.skill_name} ${card.days_overdue > 0 ? `<span class="overdue-badge">quá hạn ${card.days_overdue} ngày</span>` : ''}</div>
            <div class="quiz-q-text" id="review-q-text">${q.question_latex}</div>
            <div class="quiz-options" id="review-options">
                ${q.choices.map((c, i) => `
                    <button class="quiz-option-btn" data-index="${i}" onclick="selectReviewOption(${i}, this)">
                        <span class="quiz-option-label">${['A', 'B', 'C', 'D'][i]}</span>
                        <span class="quiz-option-text" id="review-opt-${i}">${c}</span>
                    </button>
                `).join('')}
            </div>
            <button class="quiz-submit-btn" id="review-submit" disabled>Xác nhận</button>
            <div class="quiz-feedback" id="review-feedback" style="display:none"></div>
        </div>
    `;

    renderKatex(document.getElementById('review-q-text'));
    q.choices.forEach((_, i) => {
        const el = document.getElementById(`review-opt-${i}`);
        if (el) renderKatex(el);
    });

    let reviewSelected = -1;
    document.getElementById('review-submit').onclick = async () => {
        if (reviewSelected < 0) return;
        const submitBtn = document.getElementById('review-submit');
        submitBtn.disabled = true;
        submitBtn.textContent = '⏳...';

        // Submit review
        try {
            const res = await fetch(`${API}/review/submit`, {
                method: 'POST',
                headers: authHeaders(),
                body: JSON.stringify({ card_id: card.card_id, question_id: q.id, selected_index: reviewSelected }),
            });
            const data = await res.json();

            const isCorrect = data.is_correct;

            // Highlight
            document.querySelectorAll('#review-options .quiz-option-btn').forEach((btn, i) => {
                btn.disabled = true;
                btn.onclick = null;
                if (i === data.correct_index) btn.classList.add('correct');
                if (i === reviewSelected && !isCorrect) btn.classList.add('wrong');
            });

            const fb = document.getElementById('review-feedback');
            fb.style.display = 'block';
            fb.className = `quiz-feedback ${isCorrect ? 'correct' : 'wrong'}`;
            fb.innerHTML = `<strong>${isCorrect ? '✅ Nhớ rồi!' : '❌ Cần ôn lại!'}</strong>
                <p>${data.message}</p>
                ${data.explanation ? `<div class="quiz-explanation">${data.explanation}</div>` : ''}`;
            renderKatex(fb);

            submitBtn.textContent = index + 1 < cards.length ? 'Thẻ tiếp →' : 'Hoàn thành';
            submitBtn.disabled = false;
            submitBtn.onclick = () => renderReviewCard(cards, index + 1);
        } catch (e) {
            submitBtn.textContent = '⚠️ Lỗi';
            submitBtn.disabled = false;
        }
    };

    // Wire up option selection
    document.querySelectorAll('#review-options .quiz-option-btn').forEach(btn => {
        btn.addEventListener('click', function () {
            document.querySelectorAll('#review-options .quiz-option-btn').forEach(b => b.classList.remove('selected'));
            this.classList.add('selected');
            reviewSelected = parseInt(this.dataset.index);
            document.getElementById('review-submit').disabled = false;
        });
    });
}

function selectReviewOption(idx, btn) {
    // handled by event listener above
}


// ═══════════════════════════════════════════════════
//  EXAM PRACTICE (pre-built exam sets)
// ═══════════════════════════════════════════════════

let examState = {
    examId: null,
    questions: [],
    currentIndex: 0,
    answers: {},       // { question_id: { mcq_answer, tf_answers, sa_answer } }
    examSource: '',
    year: 0,
};

async function showExamSelector() {
    const modal = document.getElementById('exam-modal');
    const list = document.getElementById('exam-list');
    list.innerHTML = '<p style="color:#888;text-align:center;padding:20px">Đang tải danh sách đề...</p>';
    modal.style.display = 'flex';

    try {
        const res = await fetch(`${API}/exams`, { headers: authHeaders() });
        const data = await res.json();
        list.innerHTML = '';

        if (!data.exams || data.exams.length === 0) {
            list.innerHTML = '<p style="color:#888;text-align:center;padding:20px">Chưa có đề thi nào.</p>';
            return;
        }

        for (const exam of data.exams) {
            const card = document.createElement('div');
            card.className = 'exam-card';

            const typeSummary = [];
            if (exam.type_counts.exam_mcq) typeSummary.push(`${exam.type_counts.exam_mcq} TN`);
            if (exam.type_counts.exam_true_false) typeSummary.push(`${exam.type_counts.exam_true_false} ĐS`);
            if (exam.type_counts.exam_short_answer) typeSummary.push(`${exam.type_counts.exam_short_answer} TLN`);

            card.innerHTML = `
                <div class="exam-card-info">
                    <div class="exam-card-title">${esc(exam.exam_source)}</div>
                    <div class="exam-card-meta">
                        <span class="exam-card-year">${exam.year || ''}</span>
                        <span class="exam-card-count">${exam.question_count} câu</span>
                        <span class="exam-card-types">${typeSummary.join(' · ')}</span>
                    </div>
                </div>
                <button class="exam-card-btn" onclick="startExam('${exam.exam_id}')">Làm bài →</button>
            `;
            list.appendChild(card);
        }
    } catch (e) {
        list.innerHTML = '<p style="color:#f87171;text-align:center">Không thể tải danh sách đề thi.</p>';
    }
}

function closeExamModal() {
    document.getElementById('exam-modal').style.display = 'none';
}

async function startExam(examId) {
    closeExamModal();
    const welcome = document.getElementById('welcome-msg');
    if (welcome) welcome.style.display = 'none';

    const panel = document.getElementById('quiz-panel');
    const qArea = document.getElementById('quiz-question-area');
    const rArea = document.getElementById('quiz-result-area');

    panel.style.display = 'block';
    rArea.style.display = 'none';
    qArea.innerHTML = '<p class="quiz-loading">📄 Đang tải đề thi...</p>';
    document.getElementById('quiz-title').textContent = '📄 Đề Thi Thử';

    try {
        const res = await fetch(`${API}/exams/${examId}`, { headers: authHeaders() });
        if (!res.ok) throw new Error('Không tìm thấy đề thi');
        const data = await res.json();

        examState.examId = examId;
        examState.questions = data.questions;
        examState.currentIndex = 0;
        examState.answers = {};
        examState.examSource = data.exam_source;
        examState.year = data.year;

        document.getElementById('quiz-title').textContent = `📄 ${data.exam_source} ${data.year || ''}`;
        document.getElementById('quiz-score-text').textContent = `Câu 1/${data.question_count}`;
        document.getElementById('quiz-correct-text').textContent = `0/${data.question_count} đã trả lời`;

        renderExamQuestion(0);
    } catch (e) {
        qArea.innerHTML = `<p class="quiz-error">⚠️ ${e.message}</p>`;
    }
}

function renderExamQuestion(index) {
    const q = examState.questions[index];
    if (!q) return;
    const total = examState.questions.length;
    const qArea = document.getElementById('quiz-question-area');
    examState.currentIndex = index;

    const pct = ((index) / total) * 100;
    document.getElementById('quiz-progress-fill').style.width = pct + '%';
    document.getElementById('quiz-score-text').textContent = `Câu ${index + 1}/${total}`;

    // Count answered
    const answeredCount = Object.keys(examState.answers).length;
    document.getElementById('quiz-correct-text').textContent = `${answeredCount}/${total} đã trả lời`;

    const partLabel = { 1: 'Phần 1 — Trắc nghiệm', 2: 'Phần 2 — Đúng/Sai', 3: 'Phần 3 — Trả lời ngắn' };
    const typeClass = { exam_mcq: 'mcq', exam_true_false: 'true_false', exam_short_answer: 'short_answer' };

    let bodyHTML = '';
    if (q.type === 'exam_mcq') {
        bodyHTML = renderExamMCQ(q, index);
    } else if (q.type === 'exam_true_false') {
        bodyHTML = renderExamTF(q, index);
    } else if (q.type === 'exam_short_answer') {
        bodyHTML = renderExamSA(q, index);
    }

    // Image HTML — for TF questions, only show "unclaimed" question-level images at top
    let imageHTML = '';
    if (q.has_image && q.image_path) {
        if (q.type === 'exam_true_false' && q.question_images) {
            // Only show images not assigned to specific statements
            const qImgs = q.question_images || [];
            if (qImgs.length > 0) {
                imageHTML = qImgs.map(p => `<img class="exam-q-image" src="/${p}" alt="Hình câu ${q.question_number}" onclick="openImageLightbox('/${p}')" style="cursor:pointer" onerror="this.style.display='none'">`).join('');
            }
        } else {
            // MCQ, SA: show all images at top
            const paths = q.image_path.split(',').map(p => p.trim()).filter(p => p);
            imageHTML = paths.map(p => `<img class="exam-q-image" src="/${p}" alt="Hình câu ${q.question_number}" onclick="openImageLightbox('/${p}')" style="cursor:pointer" onerror="this.style.display='none'">`).join('');
        }
    }

    // Navigation buttons
    const prevBtn = index > 0
        ? `<button class="exam-nav-btn" onclick="renderExamQuestion(${index - 1})">← Câu trước</button>`
        : `<button class="exam-nav-btn" disabled>← Câu trước</button>`;
    const nextBtn = index < total - 1
        ? `<button class="exam-nav-btn" onclick="renderExamQuestion(${index + 1})">Câu sau →</button>`
        : `<button class="exam-nav-btn" disabled>Câu sau →</button>`;
    const submitBtn = `<button class="exam-submit-all-btn" onclick="confirmSubmitExam()">📮 Nộp bài (${answeredCount}/${total})</button>`;

    // Question number navigation
    let qNavHTML = '<div class="exam-q-nav">';
    for (let i = 0; i < total; i++) {
        const isAnswered = examState.answers[examState.questions[i].id] !== undefined;
        const isCurrent = i === index;
        qNavHTML += `<button class="exam-q-nav-btn ${isCurrent ? 'current' : ''} ${isAnswered ? 'answered' : ''}" onclick="renderExamQuestion(${i})">${i + 1}</button>`;
    }
    qNavHTML += '</div>';

    qArea.innerHTML = `
        <div class="quiz-question-card exam-question-card">
            ${qNavHTML}
            <div class="quiz-q-number">
                <span class="quiz-type-badge type-${typeClass[q.type] || 'mcq'}">${partLabel[q.difficulty_part] || ''}</span>
                Câu ${index + 1}/${total} · ${q.chapter || ''}
            </div>
            <div class="quiz-q-text" id="exam-q-text-${index}">${formatText(q.question_text || '')}</div>
            ${imageHTML ? `<div class="exam-images">${imageHTML}</div>` : ''}
            ${bodyHTML}
            <div class="exam-nav-bar">
                ${prevBtn}
                ${submitBtn}
                ${nextBtn}
            </div>
        </div>
    `;

    renderKatex(document.getElementById(`exam-q-text-${index}`));

    // Render KaTeX in options
    if (q.type === 'exam_mcq' && q.choices) {
        q.choices.forEach((_, i) => {
            const el = document.getElementById(`exam-opt-${index}-${i}`);
            if (el) renderKatex(el);
        });
    }
    if (q.type === 'exam_true_false' && q.statements) {
        q.statements.forEach((_, i) => {
            const el = document.getElementById(`exam-stmt-${index}-${i}`);
            if (el) renderKatex(el);
        });
    }
}

function renderExamMCQ(q, index) {
    const saved = examState.answers[q.id];
    const savedAnswer = saved ? saved.mcq_answer : null;

    return `
        <div class="quiz-options" id="exam-options-${index}">
            ${(q.choices || []).map((c, i) => {
        const label = ['A', 'B', 'C', 'D'][i];
        const isSelected = savedAnswer === label;
        return `
                    <button class="quiz-option-btn ${isSelected ? 'selected' : ''}" data-index="${i}" onclick="selectExamMCQ('${q.id}', ${index}, ${i}, this)">
                        <span class="quiz-option-label">${label}</span>
                        <span class="quiz-option-text" id="exam-opt-${index}-${i}">${c}</span>
                    </button>
                `;
    }).join('')}
        </div>
    `;
}

function renderExamTF(q, index) {
    const saved = examState.answers[q.id];
    const savedTF = saved ? saved.tf_answers : {};
    const stmtImages = q.statement_images || {};

    return `
        <div class="tf-statements" id="exam-tf-${index}">
            ${(q.statements || []).map((s, i) => {
        const label = ['a', 'b', 'c', 'd'][i];
        const savedVal = savedTF[label];
        const imgPath = stmtImages[label];
        const imgBtn = imgPath
            ? `<button class="stmt-view-img-btn" onclick="openImageLightbox('/${imgPath}')" title="Xem hình minh họa">📷 Xem hình</button>`
            : '';
        return `
                    <div class="tf-statement-row" id="exam-tf-row-${index}-${i}">
                        <div class="tf-statement-content">
                            <div class="tf-statement-text" id="exam-stmt-${index}-${i}">
                                <span class="tf-label">${label})</span>
                                ${s.text || s}
                            </div>
                            ${imgBtn}
                        </div>
                        <div class="tf-toggle-group">
                            <button class="tf-toggle-btn ${savedVal === true ? 'active' : ''}" data-val="true" onclick="selectExamTF('${q.id}', ${index}, '${label}', true, this)">
                                Đúng
                            </button>
                            <button class="tf-toggle-btn ${savedVal === false ? 'active' : ''}" data-val="false" onclick="selectExamTF('${q.id}', ${index}, '${label}', false, this)">
                                Sai
                            </button>
                        </div>
                    </div>
                `;
    }).join('')}
        </div>
    `;
}

function renderExamSA(q, index) {
    const saved = examState.answers[q.id];
    const savedVal = saved ? saved.sa_answer : '';

    return `
        <div class="sa-input-area" id="exam-sa-${index}">
            <input type="text" class="sa-input" id="exam-sa-input-${index}"
                   placeholder="Nhập đáp án..."
                   value="${esc(savedVal)}"
                   oninput="onExamSAInput('${q.id}', ${index})">
        </div>
    `;
}

// ── Exam selection handlers ──

function selectExamMCQ(qId, qIndex, optIndex, btn) {
    const optionsDiv = document.getElementById(`exam-options-${qIndex}`);
    optionsDiv.querySelectorAll('.quiz-option-btn').forEach(b => b.classList.remove('selected'));
    btn.classList.add('selected');

    const label = ['A', 'B', 'C', 'D'][optIndex];
    examState.answers[qId] = { mcq_answer: label };
    updateExamAnswerCount();
}

function selectExamTF(qId, qIndex, label, value, btn) {
    const row = btn.closest('.tf-statement-row');
    row.querySelectorAll('.tf-toggle-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');

    if (!examState.answers[qId]) {
        examState.answers[qId] = { tf_answers: {} };
    }
    examState.answers[qId].tf_answers[label] = value;
    updateExamAnswerCount();
}

function onExamSAInput(qId, qIndex) {
    const val = document.getElementById(`exam-sa-input-${qIndex}`).value.trim();
    if (val) {
        examState.answers[qId] = { sa_answer: val };
    } else {
        delete examState.answers[qId];
    }
    updateExamAnswerCount();
}

function updateExamAnswerCount() {
    const total = examState.questions.length;
    const answered = Object.keys(examState.answers).length;
    document.getElementById('quiz-correct-text').textContent = `${answered}/${total} đã trả lời`;

    // Update nav buttons
    document.querySelectorAll('.exam-q-nav-btn').forEach((btn, i) => {
        const qId = examState.questions[i]?.id;
        if (qId && examState.answers[qId]) {
            btn.classList.add('answered');
        } else {
            btn.classList.remove('answered');
        }
    });

    // Update submit button text
    const submitBtn = document.querySelector('.exam-submit-all-btn');
    if (submitBtn) submitBtn.textContent = `📮 Nộp bài (${answered}/${total})`;
}

function confirmSubmitExam() {
    const total = examState.questions.length;
    const answered = Object.keys(examState.answers).length;
    const unanswered = total - answered;

    let msg = `Bạn đã trả lời ${answered}/${total} câu.`;
    if (unanswered > 0) {
        msg += `\n\n⚠️ Còn ${unanswered} câu chưa trả lời. Những câu này sẽ được tính 0 điểm.`;
    }
    msg += '\n\nXác nhận nộp bài?';

    if (confirm(msg)) {
        submitExam();
    }
}

async function submitExam() {
    const qArea = document.getElementById('quiz-question-area');
    const rArea = document.getElementById('quiz-result-area');
    qArea.style.display = 'none';
    rArea.style.display = 'block';
    rArea.innerHTML = '<p class="quiz-loading">⏳ Đang chấm điểm...</p>';
    document.getElementById('quiz-progress-fill').style.width = '100%';

    try {
        const res = await fetch(`${API}/exams/grade`, {
            method: 'POST',
            headers: authHeaders(),
            body: JSON.stringify({
                exam_id: examState.examId,
                answers: examState.answers,
            }),
        });
        if (!res.ok) throw new Error('Lỗi chấm điểm');
        const data = await res.json();

        renderExamResult(data);
    } catch (e) {
        rArea.innerHTML = `<p class="quiz-error">⚠️ ${e.message}</p>`;
    }
}

function renderExamResult(data) {
    const rArea = document.getElementById('quiz-result-area');
    const scorePct = data.score_percent;
    const emoji = scorePct >= 80 ? '🎉' : scorePct >= 60 ? '👍' : scorePct >= 40 ? '📝' : '💪';

    // Part scores
    const partLabels = { mcq: 'Phần 1 (Trắc nghiệm)', true_false: 'Phần 2 (Đúng/Sai)', short_answer: 'Phần 3 (Trả lời ngắn)' };
    let partScoresHTML = '';
    if (data.part_scores) {
        partScoresHTML = '<div class="part-scores">';
        for (const [key, val] of Object.entries(data.part_scores)) {
            partScoresHTML += `
                <div class="part-score-item">
                    <span class="part-label">${partLabels[key] || key}:</span>
                    <span class="part-value">${val}đ</span>
                </div>
            `;
        }
        partScoresHTML += '</div>';
    }

    // Detailed results
    let detailHTML = '<div class="exam-result-details">';
    detailHTML += '<h4 style="margin:16px 0 8px">Chi tiết từng câu:</h4>';

    for (const r of data.results) {
        const partName = { 1: 'TN', 2: 'ĐS', 3: 'TLN' }[r.difficulty_part] || '';
        const statusIcon = r.is_correct ? '✅' : (r.points_earned > 0 ? '🟡' : '❌');
        const ptsText = `${r.points_earned}/${r.points_max}đ`;

        let answerDetail = '';
        if (r.type === 'exam_mcq') {
            answerDetail = `Bạn chọn: <strong>${r.student_answer || '—'}</strong> · Đáp án: <strong>${r.correct_answer}</strong>`;
        } else if (r.type === 'exam_true_false') {
            const details = (r.statement_details || []).map(d => {
                const mark = d.is_correct ? '✅' : '❌';
                const studentText = d.student === true ? 'Đ' : (d.student === false ? 'S' : '—');
                const correctText = d.correct ? 'Đ' : 'S';
                return `${mark} ${d.label}) ${studentText} (→${correctText})`;
            }).join(' · ');
            answerDetail = details;
        } else if (r.type === 'exam_short_answer') {
            answerDetail = `Bạn trả lời: <strong>${esc(r.student_answer) || '—'}</strong> · Đáp án: <strong>${esc(r.correct_answer)}</strong>`;
        }

        detailHTML += `
            <div class="exam-result-item ${r.is_correct ? 'correct' : (r.points_earned > 0 ? 'partial' : 'wrong')}">
                <div class="exam-result-item-header">
                    <span>${statusIcon} Câu ${r.question_number} [${partName}]</span>
                    <span class="exam-result-pts">${ptsText}</span>
                </div>
                <div class="exam-result-answer">${answerDetail}</div>
                ${r.explanation ? `<details class="exam-result-explain"><summary>Xem giải thích</summary><div class="exam-explain-content" id="explain-${r.id}">${formatText(r.explanation)}</div></details>` : ''}
            </div>
        `;
    }
    detailHTML += '</div>';

    rArea.innerHTML = `
        <div class="quiz-result-card exam-result-card">
            <div class="quiz-result-emoji">${emoji}</div>
            <div class="quiz-result-score">${data.total_score}/${data.max_score}đ</div>
            <div class="quiz-result-summary">${scorePct.toFixed(1)}% — ${data.total_questions} câu hỏi</div>
            ${partScoresHTML}
            ${detailHTML}
            <div class="quiz-result-actions">
                <button class="quiz-action-btn" onclick="closeQuizPanel()">Đóng</button>
                <button class="quiz-action-btn primary" onclick="showExamSelector()">Làm đề khác</button>
            </div>
        </div>
    `;

    // Render KaTeX in explanations
    data.results.forEach(r => {
        const el = document.getElementById(`explain-${r.id}`);
        if (el) renderKatex(el);
    });
}


// ═══════════════════════════════════════════════════
//  TEXT FORMATTING (Markdown + LaTeX)
// ═══════════════════════════════════════════════════

function formatText(text) {
    if (!text) return '';

    // ── Extract <thinking> blocks and render as collapsible ──
    let thinkingHTML = '';
    text = text.replace(/<thinking>([\s\S]*?)<\/thinking>/g, (_m, inner) => {
        const lines = inner.trim().split('\n').map(l => esc(l)).join('<br>');
        thinkingHTML += `
            <details class="thinking-block">
                <summary class="thinking-summary">🧠 AI đã suy nghĩ và kiểm chứng bài giải</summary>
                <div class="thinking-content">${lines}</div>
            </details>`;
        return '';
    });

    // ── Defense-in-depth: <answer> tag là định dạng máy (evaluation pipeline),
    // backend đã strip nhưng nếu còn sót thì render thành dòng đáp án đọc được ──
    text = text.replace(/<answer\b[^>]*>([\s\S]*?)<\/answer\s*>/gi, (_m, inner) => {
        const val = inner.trim();
        return val ? `**Đáp án:** ${val}` : '';
    });

    const stash = [];
    const ph = (s) => { stash.push(s); return `\x02LATEX${stash.length - 1}\x03`; };

    let s = text;
    s = s.replace(/\$\$([\s\S]+?)\$\$/g, (_m, inner) => ph(`$$${inner}$$`));
    s = s.replace(/\\\[([\s\S]+?)\\\]/g, (_m, inner) => ph(`\\[${inner}\\]`));
    s = s.replace(/\$([^\n$]+?)\$/g, (_m, inner) => ph(`$${inner}$`));
    s = s.replace(/\\\(([^\n]+?)\\\)/g, (_m, inner) => ph(`\\(${inner}\\)`));

    if (typeof marked !== 'undefined') {
        marked.setOptions({ breaks: true, gfm: true });
        s = marked.parse(s);
    } else {
        s = s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
        s = s.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
        s = s.replace(/\n/g, '<br>');
    }

    s = s.replace(/\x02LATEX(\d+)\x03/g, (_m, i) => {
        return stash[+i]
            .replace(/&amp;/g, '&')
            .replace(/&lt;/g, '<')
            .replace(/&gt;/g, '>');
    });

    const finalHtml = thinkingHTML + s;
    // Sanitize: nội dung LLM/OCR có thể chứa HTML thô — chặn XSS trước khi
    // gán vào innerHTML. Fallback khi CDN DOMPurify không tải được: giữ nguyên
    // (hành vi cũ) vì chat vẫn phải hiển thị được.
    if (typeof DOMPurify !== 'undefined') {
        return DOMPurify.sanitize(finalHtml);
    }
    return finalHtml;
}

function renderKatex(element) {
    const tryRender = () => {
        if (typeof renderMathInElement !== 'undefined') {
            renderMathInElement(element, {
                delimiters: [
                    { left: '$$', right: '$$', display: true },
                    { left: '\\[', right: '\\]', display: true },
                    { left: '$', right: '$', display: false },
                    { left: '\\(', right: '\\)', display: false },
                ],
                throwOnError: false,
            });
        }
    };
    if (window._katexReady) tryRender();
    else window.addEventListener('load', tryRender, { once: true });
}

function esc(s) {
    if (!s) return '';
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}


// ═══════════════════════════════════════════════════
//  IMAGE LIGHTBOX
// ═══════════════════════════════════════════════════

function openImageLightbox(src) {
    // Remove existing lightbox if any
    const existing = document.getElementById('image-lightbox');
    if (existing) existing.remove();

    const lightbox = document.createElement('div');
    lightbox.id = 'image-lightbox';
    lightbox.className = 'image-lightbox-overlay';
    lightbox.innerHTML = `
        <div class="image-lightbox-backdrop" onclick="closeImageLightbox()"></div>
        <div class="image-lightbox-content">
            <button class="image-lightbox-close" onclick="closeImageLightbox()" title="Đóng">✕</button>
            <img src="${src}" alt="Hình minh họa" class="image-lightbox-img">
        </div>
    `;
    document.body.appendChild(lightbox);

    // Animate in
    requestAnimationFrame(() => lightbox.classList.add('active'));

    // Close on Escape
    const escHandler = (e) => {
        if (e.key === 'Escape') {
            closeImageLightbox();
            document.removeEventListener('keydown', escHandler);
        }
    };
    document.addEventListener('keydown', escHandler);
}

function closeImageLightbox() {
    const lightbox = document.getElementById('image-lightbox');
    if (!lightbox) return;
    lightbox.classList.remove('active');
    setTimeout(() => lightbox.remove(), 200);
}


// ═══════════════════════════════════════════════════
//  EXAM SOLVER (OCR + Per-Question RAG Pipeline)
// ═══════════════════════════════════════════════════

let examSolverState = {
    selectedEngine: 'cloud',
    isProcessing: false,
    currentFile: null,
};

function showExamSolver() {
    const panel = document.getElementById('exam-solver-panel');
    panel.style.display = 'flex';

    // Hide other panels
    const welcome = document.getElementById('welcome-msg');
    if (welcome) welcome.style.display = 'none';
    document.getElementById('quiz-panel').style.display = 'none';

    // Hide chat elements to prevent squeezing
    document.getElementById('chat-messages').style.display = 'none';
    const inputArea = document.querySelector('.chat-input-area');
    if (inputArea) inputArea.style.display = 'none';

    // Reset to upload state
    document.getElementById('exam-solver-upload').style.display = 'flex';
    document.getElementById('exam-solver-confirm').style.display = 'none';
    document.getElementById('exam-solver-progress').style.display = 'none';
    document.getElementById('exam-solver-results').style.display = 'none';

    // Setup drag-drop
    setupExamDropzone();
}

function closeExamSolver() {
    document.getElementById('exam-solver-panel').style.display = 'none';
    
    // Restore chat elements
    document.getElementById('chat-messages').style.display = 'flex';
    const inputArea = document.querySelector('.chat-input-area');
    if (inputArea) inputArea.style.display = 'block';
}

function selectOCREngine(engine, btn) {
    examSolverState.selectedEngine = engine;
    document.querySelectorAll('.ocr-option').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');

    const hint = document.getElementById('ocr-hint');
    if (engine === 'cloud') {
        hint.textContent = 'GPT Vision — độ chính xác cao nhất cho công thức Toán';
    } else {
        hint.textContent = 'GOT-OCR2.0 local — miễn phí, offline (cần GPU để nhanh)';
    }
}

function setupExamDropzone() {
    const dropzone = document.getElementById('exam-dropzone');
    if (!dropzone || dropzone._setupDone) return;
    dropzone._setupDone = true;

    // Click to select file
    dropzone.addEventListener('click', () => {
        document.getElementById('exam-file-input').click();
    });

    // Drag & drop
    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });
    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('dragover');
    });
    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            processExamFile(files[0]);
        }
    });
}

function handleExamFileSelect(event) {
    const file = event.target.files[0];
    if (file) processExamFile(file);
}

async function processExamFile(file) {
    if (examSolverState.isProcessing) return;

    // Validate
    const validTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/gif', 'application/pdf'];
    if (!validTypes.includes(file.type)) {
        alert('Chỉ hỗ trợ file ảnh (JPEG, PNG, WebP) hoặc PDF');
        return;
    }
    if (file.size > 20 * 1024 * 1024) {
        alert('File quá lớn (tối đa 20MB)');
        return;
    }

    examSolverState.isProcessing = true;
    examSolverState.currentFile = file;

    // Switch to progress view
    document.getElementById('exam-solver-upload').style.display = 'none';
    document.getElementById('exam-solver-confirm').style.display = 'none';
    document.getElementById('exam-solver-progress').style.display = 'flex';
    document.getElementById('exam-solver-results').style.display = 'none';
    document.getElementById('exam-progress-fill').style.width = '50%';
    document.getElementById('exam-status-text').textContent = `Đang trích xuất văn bản (OCR ${examSolverState.selectedEngine})...`;
    document.getElementById('exam-progress-detail').textContent = 'Vui lòng đợi trong giây lát...';

    // Build form data
    const formData = new FormData();
    formData.append('file', file);
    formData.append('ocr_engine', examSolverState.selectedEngine);

    try {
        const ocrHeaders = {};
        if (currentUserId) ocrHeaders['X-User-Id'] = currentUserId;
        const response = await fetch(`${API}/exam-solver/ocr`, {
            method: 'POST',
            headers: ocrHeaders,
            body: formData,
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Lỗi OCR');
        }

        const data = await response.json();
        
        // Switch to confirm view
        document.getElementById('exam-solver-progress').style.display = 'none';
        document.getElementById('exam-solver-confirm').style.display = 'flex';
        document.getElementById('exam-ocr-textarea').value = data.raw_ocr || '';
        
    } catch (e) {
        document.getElementById('exam-status-text').textContent = `⚠️ Lỗi: ${e.message}`;
        document.getElementById('exam-progress-fill').style.width = '0%';
        document.getElementById('exam-progress-fill').style.background = '#f87171';
    } finally {
        examSolverState.isProcessing = false;
    }
}

async function confirmAndSolveExam() {
    if (examSolverState.isProcessing || !examSolverState.currentFile) return;

    examSolverState.isProcessing = true;

    // Switch to progress view
    document.getElementById('exam-solver-confirm').style.display = 'none';
    document.getElementById('exam-solver-progress').style.display = 'flex';
    document.getElementById('exam-progress-fill').style.width = '0%';
    document.getElementById('exam-status-text').textContent = `Đang giải đề thi...`;
    document.getElementById('exam-progress-detail').textContent = '';

    const formData = new FormData();
    formData.append('file', examSolverState.currentFile);
    formData.append('ocr_engine', examSolverState.selectedEngine);
    
    const editedText = document.getElementById('exam-ocr-textarea').value;
    formData.append('raw_ocr_text', editedText);

    try {
        const solveHeaders = {};
        if (currentUserId) solveHeaders['X-User-Id'] = currentUserId;
        const response = await fetch(`${API}/exam-solver/solve`, {
            method: 'POST',
            headers: solveHeaders,
            body: formData,
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop();

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                try {
                    const data = JSON.parse(line.slice(6));
                    handleExamSSEEvent(data);
                } catch (e) { /* ignore parse errors */ }
            }
        }
    } catch (e) {
        document.getElementById('exam-status-text').textContent = `⚠️ Lỗi: ${e.message}`;
        document.getElementById('exam-progress-fill').style.width = '0%';
        document.getElementById('exam-progress-fill').style.background = '#f87171';
    } finally {
        examSolverState.isProcessing = false;
    }
}

function handleExamSSEEvent(data) {
    switch (data.type) {
        case 'ocr':
            document.getElementById('exam-status-text').textContent = data.message;
            document.getElementById('exam-progress-fill').style.width = '5%';
            break;

        case 'progress':
            const pct = ((data.current / data.total) * 100).toFixed(0);
            document.getElementById('exam-progress-fill').style.width = `${pct}%`;
            document.getElementById('exam-status-text').textContent =
                `Đang giải ${data.question} (${data.current}/${data.total})`;
            document.getElementById('exam-progress-detail').textContent =
                `${data.current}/${data.total} câu đã giải`;
            break;

        case 'done':
            document.getElementById('exam-progress-fill').style.width = '100%';
            document.getElementById('exam-status-text').textContent = '✅ Hoàn thành!';
            setTimeout(() => renderExamSolverResult(data.result), 500);
            break;

        case 'error':
            document.getElementById('exam-status-text').textContent = `⚠️ Lỗi: ${data.message}`;
            document.getElementById('exam-progress-fill').style.background = '#f87171';
            break;
    }
}

function renderExamSolverResult(result) {
    document.getElementById('exam-solver-progress').style.display = 'none';
    const resultsDiv = document.getElementById('exam-solver-results');
    resultsDiv.style.display = 'block';

    // Summary header
    let html = `
        <div class="exam-solver-summary">
            <div class="exam-solver-summary-stats">
                <div class="solver-stat">
                    <span class="solver-stat-value">${result.total_questions}</span>
                    <span class="solver-stat-label">Câu hỏi</span>
                </div>
                <div class="solver-stat">
                    <span class="solver-stat-value">${result.elapsed_seconds.toFixed(1)}s</span>
                    <span class="solver-stat-label">Thời gian</span>
                </div>
                <div class="solver-stat">
                    <span class="solver-stat-value">${result.ocr_engine_used === 'cloud' ? '☁️' : '🖥️'}</span>
                    <span class="solver-stat-label">${result.ocr_engine_used === 'cloud' ? 'Cloud OCR' : 'Local OCR'}</span>
                </div>
            </div>
            <div class="exam-solver-actions">
                <button class="solver-action-btn" onclick="toggleRawOCR()">📋 Xem OCR gốc</button>
                <button class="solver-action-btn primary" onclick="resetExamSolver()">🔄 Giải đề khác</button>
            </div>
        </div>

        <div class="exam-solver-raw-ocr" id="exam-solver-raw-ocr" style="display:none">
            <h4>📋 OCR Text gốc:</h4>
            <pre>${esc(result.raw_ocr)}</pre>
        </div>
    `;

    // Per-question solutions
    html += '<div class="exam-solver-questions">';
    for (const q of result.questions) {
        const skillBadge = q.skill_name
            ? `<span class="solver-skill-badge">${esc(q.skill_name)}</span>`
            : '';
        const typeBadge = {
            mcq: '📝 Trắc nghiệm',
            true_false: '✅ Đúng/Sai',
            short_answer: '✏️ Trả lời ngắn',
            essay: '📄 Tự luận',
        }[q.question_type] || q.question_type;

        const errorClass = q.error ? 'has-error' : '';

        html += `
            <div class="solver-question-card ${errorClass}">
                <div class="solver-q-header">
                    <span class="solver-q-number">${esc(q.question_number)}</span>
                    <span class="solver-q-type">${typeBadge}</span>
                    ${skillBadge}
                </div>
                <details class="solver-q-problem" open>
                    <summary>Đề bài</summary>
                    <div class="solver-q-content" id="solver-q-${result.questions.indexOf(q)}">${formatText(q.content)}</div>
                </details>
                <div class="solver-q-solution">
                    <h4>💡 Lời giải:</h4>
                    <div class="solver-solution-content" id="solver-sol-${result.questions.indexOf(q)}">${formatText(q.solution)}</div>
                </div>
                ${q.error ? `<div class="solver-q-error">⚠️ ${esc(q.error)}</div>` : ''}
            </div>
        `;
    }
    html += '</div>';

    resultsDiv.innerHTML = html;

    // Render KaTeX in all question/solution elements
    result.questions.forEach((_, i) => {
        const qEl = document.getElementById(`solver-q-${i}`);
        const sEl = document.getElementById(`solver-sol-${i}`);
        if (qEl) renderKatex(qEl);
        if (sEl) renderKatex(sEl);
    });
}

function toggleRawOCR() {
    const el = document.getElementById('exam-solver-raw-ocr');
    el.style.display = el.style.display === 'none' ? 'block' : 'none';
}

function resetExamSolver() {
    document.getElementById('exam-solver-upload').style.display = 'flex';
    document.getElementById('exam-solver-confirm').style.display = 'none';
    document.getElementById('exam-solver-progress').style.display = 'none';
    document.getElementById('exam-solver-results').style.display = 'none';
    document.getElementById('exam-progress-fill').style.width = '0%';
    document.getElementById('exam-progress-fill').style.background = '';
    document.getElementById('exam-file-input').value = '';
}


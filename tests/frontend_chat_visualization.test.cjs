const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');


function createHarness() {
    const elements = new Map();

    class FakeElement {
        constructor(tagName = 'div') {
            this.tagName = tagName;
            this.children = [];
            this.className = '';
            this.classList = {
                add: (...names) => {
                    this.className = [this.className, ...names].filter(Boolean).join(' ');
                },
                remove: () => {},
                toggle: () => {},
            };
            this.style = {};
            this.scrollHeight = 0;
            this.textContent = '';
            this._innerHTML = '';
            this._msgBody = null;
        }

        set id(value) {
            this._id = value;
            elements.set(value, this);
        }

        get id() {
            return this._id;
        }

        set innerHTML(value) {
            this._innerHTML = value;
            const idMatch = value.match(/id="([^"]+)"/);
            if (idMatch) {
                const body = new FakeElement('div');
                body.id = idMatch[1];
                this._msgBody = body;
            } else if (value.includes('class="msg-body')) {
                this._msgBody = new FakeElement('div');
            }
        }

        get innerHTML() {
            return this._innerHTML;
        }

        appendChild(child) {
            this.children.push(child);
            this.scrollHeight = this.children.length;
            return child;
        }

        prepend(child) {
            this.children.unshift(child);
        }

        querySelector(selector) {
            if (selector === '.msg-body') return this._msgBody;
            return null;
        }

        remove() {
            this.removed = true;
        }
    }

    const chatMessages = new FakeElement('div');
    chatMessages.id = 'chat-messages';

    const document = {
        addEventListener: () => {},
        createElement: tagName => new FakeElement(tagName),
        getElementById: id => elements.get(id) || null,
        querySelector: () => null,
        querySelectorAll: () => [],
    };

    const plotCalls = [];
    const context = vm.createContext({
        TextDecoder,
        TextEncoder,
        URL,
        clearTimeout,
        console,
        document,
        fetch: null,
        localStorage: {
            getItem: () => null,
            removeItem: () => {},
            setItem: () => {},
        },
        Plotly: {
            newPlot: (...args) => {
                plotCalls.push(args);
                return Promise.resolve();
            },
        },
        Promise,
        setTimeout,
        window: {
            _katexReady: true,
            addEventListener: () => {},
        },
    });

    const appPath = path.join(__dirname, '..', 'static', 'app.js');
    vm.runInContext(fs.readFileSync(appPath, 'utf8'), context, { filename: appPath });

    return { chatMessages, context, plotCalls };
}


function streamResponse(frames) {
    const payload = frames.map(frame => `data: ${JSON.stringify(frame)}\n\n`).join('');
    let sent = false;
    return {
        ok: true,
        headers: { get: () => '42' },
        body: {
            getReader: () => ({
                read: async () => {
                    if (sent) return { done: true, value: undefined };
                    sent = true;
                    return { done: false, value: new TextEncoder().encode(payload) };
                },
            }),
        },
    };
}


test('SSE done frame renders its Plotly visualization', async () => {
    const { context, plotCalls } = createHarness();
    const visualization = {
        vis_type: 'plot',
        data: {
            traces: [{ x: [-1, 0, 1], y: [1, 0, 1], type: 'scatter' }],
            layout: { title: 'y = x²' },
        },
    };
    context.fetch = async () => streamResponse([
        { type: 'meta', session_id: 42 },
        { type: 'token', content: 'Đồ thị đã được tạo!' },
        {
            type: 'done',
            full_response: 'Đồ thị đã được tạo!',
            visualization,
        },
    ]);

    await context.sendMessageStream('vẽ đồ thị hàm số đó');

    assert.equal(plotCalls.length, 1);
    assert.deepEqual(Array.from(plotCalls[0][1][0].y), [1, 0, 1]);
});


test('missing Plotly library shows an error instead of a blank chart', () => {
    const { chatMessages, context } = createHarness();
    delete context.Plotly;

    context.renderVisualization({
        vis_type: 'plot',
        data: { traces: [{ x: [0], y: [0] }], layout: {} },
    });

    const wrapper = chatMessages.children.at(-1);
    const chart = wrapper._msgBody.children[0];
    assert.match(chart.textContent, /Không tải được thư viện/);
});


test('loading chat history restores a saved visualization', async () => {
    const { context, plotCalls } = createHarness();
    context.fetch = async () => ({
        ok: true,
        json: async () => ({
            messages: [
                { role: 'user', content: 'vẽ đồ thị hàm số đó' },
                {
                    role: 'assistant',
                    content: 'Đồ thị đã được tạo!',
                    visualization: {
                        vis_type: 'plot',
                        data: {
                            traces: [{ x: [0, 1], y: [0, 1] }],
                            layout: {},
                        },
                    },
                },
            ],
        }),
    });

    await context.loadSessionHistory(7);

    assert.equal(plotCalls.length, 1);
});

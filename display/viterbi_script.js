document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Elements ---
    const svg = document.getElementById('trellis-diagram');
    const infoDisplay = document.getElementById('info-display');
    const inputSequenceEl = document.getElementById('input-sequence');
    const encodedSequenceEl = document.getElementById('encoded-sequence');
    const startButton = document.getElementById('start-button');
    const editButton = document.getElementById('edit-button');
    const randomErrorButton = document.getElementById('random-error-button');
    const clearErrorButton = document.getElementById('clear-error-button');
    const encodingModeEl = document.getElementById('encoding-mode');
    const decodingModeEl = document.getElementById('decoding-mode');

    // --- Constants and State ---
    const NS = 'http://www.w3.org/2000/svg';
    const states = { 'S0': '00', 'S1': '01', 'S2': '10', 'S3': '11' };
    const stateNames = ['S0', 'S1', 'S2', 'S3'];
    const transitions = {
        'S0': { '0': { nextState: 'S0', output: '00' }, '1': { nextState: 'S2', output: '11' } },
        'S1': { '0': { nextState: 'S0', output: '11' }, '1': { nextState: 'S2', output: '00' } },
        'S2': { '0': { nextState: 'S1', output: '10' }, '1': { nextState: 'S3', output: '01' } },
        'S3': { '0': { nextState: 'S1', output: '01' }, '1': { nextState: 'S3', output: '10' } },
    };
    const statePositions = { 'S0': 0, 'S1': 1, 'S2': 2, 'S3': 3 }; // Y-index

    let appState = {
        inputSequence: '10101',
        encodedSequence: '',
        corruptedSequence: '',
        currentMode: 'encoding', // 'encoding' or 'decoding'
        animationRunning: false,
        animationTimer: null,
        pathMetricsHistory: [],
        survivorPathsHistory: [],
    };

    // --- Drawing Functions ---
    function getCoords(step, state) {
        const totalSteps = appState.inputSequence.length;
        const padding = 50;
        const svgWidth = svg.clientWidth;
        const svgHeight = svg.clientHeight;
        const x = padding + (svgWidth - 2 * padding) * (step / totalSteps);
        const y = padding + (svgHeight - 2 * padding) * (statePositions[state] / 3);
        return { x, y };
    }

    function drawTrellisDiagram() {
        svg.innerHTML = ''; // Clear previous diagram
        const numSteps = appState.inputSequence.length;

        // 1. Draw transition lines and labels
        for (let step = 0; step < numSteps; step++) {
            for (const state of stateNames) {
                for (const input of ['0', '1']) {
                    const trans = transitions[state][input];
                    const { x: x1, y: y1 } = getCoords(step, state);
                    const { x: x2, y: y2 } = getCoords(step + 1, trans.nextState);

                    const line = document.createElementNS(NS, 'line');
                    line.setAttribute('x1', x1); line.setAttribute('y1', y1);
                    line.setAttribute('x2', x2); line.setAttribute('y2', y2);
                    line.classList.add('transition-line', `input-${input}`, 'inactive');
                    line.id = `line-${state}-${step}-${input}`;
                    svg.appendChild(line);

                    const text = document.createElementNS(NS, 'text');
                    text.setAttribute('x', (x1 + x2) / 2 + 5);
                    text.setAttribute('y', (y1 + y2) / 2 - 5);
                    text.textContent = trans.output;
                    text.classList.add('transition-text');
                    text.id = `text-${state}-${step}-${input}`;
                    svg.appendChild(text);
                }
            }
        }

        // 2. Draw state nodes
        for (let step = 0; step <= numSteps; step++) {
            for (const state of stateNames) {
                const { x, y } = getCoords(step, state);
                const circle = document.createElementNS(NS, 'circle');
                circle.setAttribute('cx', x); circle.setAttribute('cy', y);
                circle.setAttribute('r', 15);
                circle.classList.add('state-circle');
                circle.id = `circle-${state}-${step}`;
                svg.appendChild(circle);

                const text = document.createElementNS(NS, 'text');
                text.setAttribute('x', x); text.setAttribute('y', y);
                text.textContent = state;
                text.classList.add('state-text');
                text.id = `text-${state}-${step}`;
                svg.appendChild(text);
            }
        }
    }

    // --- Core Logic ---
    function performEncoding(sequence) {
        let currentState = 'S0';
        let encoded = '';
        for (const bit of sequence) {
            const trans = transitions[currentState][bit];
            encoded += trans.output;
            currentState = trans.nextState;
        }
        return encoded;
    }

    function precomputeViterbiPaths() {
        let pathMetrics = { S0: 0, S1: Infinity, S2: Infinity, S3: Infinity };
        let survivorPaths = { S0: [], S1: [], S2: [], S3: [] };

        appState.pathMetricsHistory = [pathMetrics];
        appState.survivorPathsHistory = [survivorPaths];

        const receivedGroups = appState.corruptedSequence.match(/.{1,2}/g) || [];

        for (let t = 0; t < receivedGroups.length; t++) {
            const received = receivedGroups[t];
            const newPathMetrics = { S0: Infinity, S1: Infinity, S2: Infinity, S3: Infinity };
            const newSurvivorPaths = { S0: [], S1: [], S2: [], S3: [] };

            for (const state of stateNames) {
                if (pathMetrics[state] === Infinity) continue;

                for (const input of ['0', '1']) {
                    const trans = transitions[state][input];
                    const hammingDist = (received[0] !== trans.output[0]) + (received[1] !== trans.output[1]);
                    const newMetric = pathMetrics[state] + hammingDist;

                    if (newMetric < newPathMetrics[trans.nextState]) {
                        newPathMetrics[trans.nextState] = newMetric;
                        newSurvivorPaths[trans.nextState] = [...survivorPaths[state], { from: state, to: trans.nextState, input }];
                    }
                }
            }
            pathMetrics = newPathMetrics;
            survivorPaths = newSurvivorPaths;
            appState.pathMetricsHistory.push(pathMetrics);
            appState.survivorPathsHistory.push(survivorPaths);
        }
    }

    function viterbiDecode() {
        const lastMetrics = appState.pathMetricsHistory[appState.pathMetricsHistory.length - 1];
        let bestState = 'S0';
        let minMetric = Infinity;
        for(const state in lastMetrics) {
            if(lastMetrics[state] < minMetric) {
                minMetric = lastMetrics[state];
                bestState = state;
            }
        }
        const bestPath = appState.survivorPathsHistory[appState.survivorPathsHistory.length - 1][bestState];
        return bestPath.map(p => p.input).join('');
    }

    // --- Animation ---
    function stopAnimation() {
        if (appState.animationTimer) {
            clearTimeout(appState.animationTimer);
            appState.animationTimer = null;
        }
        appState.animationRunning = false;
        startButton.disabled = false;
        startButton.textContent = appState.currentMode === 'encoding' ? '开始编码' : '开始解码';
    }

    function animate() {
        let step = 0;
        const totalSteps = appState.inputSequence.length;
        stopAnimation();
        appState.animationRunning = true;
        startButton.disabled = true;
        startButton.textContent = '动画进行中...';
        resetAllElements();

        function frame() {
            if (appState.currentMode === 'encoding') {
                animateEncodingFrame(step);
            } else {
                animateDecodingFrame(step);
            }

            step++;
            if (step > totalSteps) {
                stopAnimation();
                if (appState.currentMode === 'decoding') {
                    const decoded = viterbiDecode();
                    infoDisplay.textContent = `解码完成！原始序列: ${appState.inputSequence}, 解码结果: ${decoded}`;
                    highlightFinalPath();
                }
                return;
            }
            appState.animationTimer = setTimeout(frame, 1000);
        }
        frame();
    }

    function animateEncodingFrame(step) {
        if (step === 0) {
            updateElement('circle-S0-0', 'active', true);
            updateElement('text-S0-0', 'active', true);
            infoDisplay.textContent = '开始于状态 S0';
            return;
        }

        let currentState = 'S0';
        let path = [];
        for (let i = 0; i < step; i++) {
            const input = appState.inputSequence[i];
            const trans = transitions[currentState][input];
            path.push({ from: currentState, to: trans.nextState, input });
            currentState = trans.nextState;
        }

        resetAllElements();
        updateElement('circle-S0-0', 'active', true);
        updateElement('text-S0-0', 'active', true);

        let currentPathState = 'S0';
        for(let i=0; i<path.length; i++){
            const p = path[i];
            updateElement(`line-${p.from}-${i}-${p.input}`, 'active', true);
            updateElement(`text-${p.from}-${i}-${p.input}`, 'active', true);
            updateElement(`circle-${p.to}-${i+1}`, 'active', true);
            updateElement(`text-${p.to}-${i+1}`, 'active', true);
            currentPathState = p.to;
        }
        const currentOutput = performEncoding(appState.inputSequence.substring(0, step));
        infoDisplay.textContent = `步骤 ${step}: 输入 '${appState.inputSequence[step-1]}' -> 输出 '${currentOutput.slice(-2)}'`;

        if (step === appState.inputSequence.length) {
            infoDisplay.textContent = `编码完成！最终输出: ${appState.encodedSequence}`;
        }
    }

    function animateDecodingFrame(step) {
        resetAllElements();
        for (let t = 0; t <= step; t++) {
            const metrics = appState.pathMetricsHistory[t];
            for (const state of stateNames) {
                if (metrics[state] !== Infinity) {
                    updateElement(`circle-${state}-${t}`, 'active', true);
                    updateElement(`text-${state}-${t}`, 'active', true);
                }
            }

            if (t > 0) {
                const paths = appState.survivorPathsHistory[t];
                for (const state of stateNames) {
                    const path = paths[state];
                    if (path && path.length > 0) {
                        const lastSegment = path[path.length - 1];
                        updateElement(`line-${lastSegment.from}-${t-1}-${lastSegment.input}`, 'active', true);
                        updateElement(`text-${lastSegment.from}-${t-1}-${lastSegment.input}`, 'active', true);
                    }
                }
            }
        }
        if (step > 0) {
             infoDisplay.textContent = `解码步骤 ${step}: 计算路径度量并更新幸存路径`;
        } else {
             infoDisplay.textContent = '解码开始于状态 S0';
        }
    }

    function highlightFinalPath(){
        const lastMetrics = appState.pathMetricsHistory[appState.pathMetricsHistory.length - 1];
        let bestState = 'S0';
        let minMetric = Infinity;
        for(const state in lastMetrics) {
            if(lastMetrics[state] < minMetric) {
                minMetric = lastMetrics[state];
                bestState = state;
            }
        }
        const bestPath = appState.survivorPathsHistory[appState.survivorPathsHistory.length - 1][bestState];
        
        resetAllElements();
        updateElement('circle-S0-0', 'active', true);
        updateElement('text-S0-0', 'active', true);

        for(let i=0; i < bestPath.length; i++){
            const p = bestPath[i];
            updateElement(`line-${p.from}-${i}-${p.input}`, 'active', true);
            updateElement(`text-${p.from}-${i}-${p.input}`, 'active', true);
            updateElement(`circle-${p.to}-${i+1}`, 'active', true);
            updateElement(`text-${p.to}-${i+1}`, 'active', true);
        }
    }

    // --- UI Helpers ---
    function updateElement(id, className, isActive) {
        const el = document.getElementById(id);
        if (el) {
            if (isActive) el.classList.add(className);
            else el.classList.remove(className);
        }
    }

    function resetAllElements() {
        document.querySelectorAll('.active').forEach(el => el.classList.remove('active'));
        document.querySelectorAll('.inactive').forEach(el => el.classList.remove('inactive'));
        document.querySelectorAll('.transition-line').forEach(el => el.classList.add('inactive'));
    }

    function updateMode() {
        appState.currentMode = encodingModeEl.checked ? 'encoding' : 'decoding';
        if (appState.currentMode === 'encoding') {
            startButton.textContent = '开始编码';
            inputSequenceEl.disabled = false;
            encodedSequenceEl.readOnly = true;
        } else {
            startButton.textContent = '开始解码';
            inputSequenceEl.disabled = true;
            encodedSequenceEl.readOnly = false;
        }
        stopAnimation();
        drawTrellisDiagram();
    }

    // --- Event Handlers ---
    function handleStart() {
        if (appState.animationRunning) return;
        appState.inputSequence = inputSequenceEl.value.replace(/[^01]/g, '');
        inputSequenceEl.value = appState.inputSequence;
        if (!appState.inputSequence) {
            alert('请输入有效的二进制序列！');
            return;
        }
        drawTrellisDiagram();

        appState.encodedSequence = performEncoding(appState.inputSequence);
        if (appState.currentMode === 'encoding') {
            appState.corruptedSequence = appState.encodedSequence;
        } else {
            appState.corruptedSequence = encodedSequenceEl.value.replace(/[^01]/g, '');
        }
        encodedSequenceEl.value = appState.corruptedSequence;

        if (appState.currentMode === 'decoding') {
            precomputeViterbiPaths();
        }
        animate();
    }

    function handleSequenceChange() {
        appState.inputSequence = inputSequenceEl.value.replace(/[^01]/g, '');
        inputSequenceEl.value = appState.inputSequence;
        if (appState.inputSequence) {
            stopAnimation();
            drawTrellisDiagram();
            infoDisplay.textContent = '准备就绪...';
        }
    }

    function handleRandomError() {
        if (!appState.encodedSequence) {
            appState.encodedSequence = performEncoding(appState.inputSequence);
        }
        let seq = appState.encodedSequence.split('');
        const len = seq.length;
        const numErrors = Math.floor(Math.random() * Math.min(3, len)) + 1;
        const errorPositions = new Set();
        while (errorPositions.size < numErrors) {
            errorPositions.add(Math.floor(Math.random() * len));
        }
        errorPositions.forEach(pos => {
            seq[pos] = seq[pos] === '0' ? '1' : '0';
        });
        appState.corruptedSequence = seq.join('');
        encodedSequenceEl.value = appState.corruptedSequence;
        infoDisplay.textContent = `在 ${[...errorPositions].join(', ')} 位置注入了 ${numErrors} 个错误。`;
        decodingModeEl.checked = true;
        updateMode();
    }

    function handleClearError() {
        appState.corruptedSequence = appState.encodedSequence;
        encodedSequenceEl.value = appState.corruptedSequence;
        infoDisplay.textContent = '错误已清除。';
    }

    function handleEdit() {
        encodedSequenceEl.readOnly = false;
        encodedSequenceEl.focus();
        decodingModeEl.checked = true;
        updateMode();
        infoDisplay.textContent = '现在可以编辑接收序列以进行解码。';
    }

    // --- Initialization ---
    function init() {
        inputSequenceEl.addEventListener('input', handleSequenceChange);
        startButton.addEventListener('click', handleStart);
        randomErrorButton.addEventListener('click', handleRandomError);
        clearErrorButton.addEventListener('click', handleClearError);
        editButton.addEventListener('click', handleEdit);
        encodingModeEl.addEventListener('change', updateMode);
        decodingModeEl.addEventListener('change', updateMode);

        // Use ResizeObserver to redraw the diagram on resize for responsiveness
        new ResizeObserver(drawTrellisDiagram).observe(svg);

        updateMode();
        drawTrellisDiagram();
    }

    init();
});


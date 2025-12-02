/**
 * AI Player for 2048 using ONNX Runtime Web
 * Loads a trained Deep Q-Network model and plays the game automatically
 */

(function () {
  'use strict';

  const AI_DELAY_MS = 100;
  const MODEL_PATH = 'model/model3.onnx';

  let aiPlaying = false;
  let session = null;
  let moveCount = 0;
  let gamesPlayed = 0;

  async function loadModel() {
    if (session) {
      return session;
    }

    try {
      console.log('Loading ONNX model from:', MODEL_PATH);
      session = await ort.InferenceSession.create(MODEL_PATH);
      console.log('✓ Model loaded successfully!');
      console.log('Input names:', session.inputNames);
      console.log('Output names:', session.outputNames);
      return session;
    } catch (err) {
      console.error('✗ Error loading model:', err);
      console.error('Make sure you have:');
      console.error('1. Trained the model using: python python/train_dqn.py');
      console.error('2. Converted it to ONNX: python python/convert_to_onnx.py');
      console.error('3. The model file exists at:', MODEL_PATH);
      return null;
    }
  }

  function getBoardState() {
    if (!window.game || !game.grid) {
      console.error('Game not ready');
      return null;
    }

    const size = game.size;
    const state = [];

    for (let x = 0; x < size; x++) {
      for (let y = 0; y < size; y++) {
        const tile = game.grid.cells[x][y];
        if (tile && tile.value > 0) {
          state.push(Math.log2(tile.value));
        } else {
          state.push(0);
        }
      }
    }

    return state;
  }

  function isGameOver() {
    if (!window.game) {
      return true;
    }
    return game.isGameTerminated();
  }

  function isMoveValid(action) {
    if (!window.game || !game.grid) {
      return false;
    }

    const oldCells = game.grid.cells.map(row => row.slice());

    const vectorMap = {
      0: { x: 0, y: -1 },   // Up
      1: { x: 1, y: 0 },    // Right
      2: { x: 0, y: 1 },    // Down
      3: { x: -1, y: 0 }    // Left
    };

    const vector = vectorMap[action];
    if (!vector) {
      return false;
    }

    for (let x = 0; x < game.size; x++) {
      for (let y = 0; y < game.size; y++) {
        const tile = game.grid.cells[x][y];
        if (tile) {
          const newX = x + vector.x;
          const newY = y + vector.y;

          if (newX >= 0 && newX < game.size && newY >= 0 && newY < game.size) {
            const targetCell = game.grid.cells[newX][newY];
            if (!targetCell || targetCell.value === tile.value) {
              return true;
            }
          }
        }
      }
    }

    return false;
  }

  async function chooseAction(state) {
    const sess = await loadModel();

    if (!sess) {
      console.warn('Model not loaded, choosing random action');
      const validActions = [];
      for (let i = 0; i < 4; i++) {
        if (isMoveValid(i)) {
          validActions.push(i);
        }
      }
      return validActions.length > 0
        ? validActions[Math.floor(Math.random() * validActions.length)]
        : 0;
    }

    const inputTensor = new ort.Tensor('float32', new Float32Array(state), [1, 16]);

    const feeds = {};
    feeds[sess.inputNames[0]] = inputTensor;
    const results = await sess.run(feeds);

    const outputTensor = results[sess.outputNames[0]];
    const qValuesArray = outputTensor.data;

    const actionScores = [];
    for (let i = 0; i < 4; i++) {
      if (isMoveValid(i)) {
        actionScores.push({ action: i, qValue: qValuesArray[i] });
      }
    }

    if (actionScores.length === 0) {
      console.warn('No valid actions available');
      return 0;
    }

    actionScores.sort((a, b) => b.qValue - a.qValue);

    const bestAction = actionScores[0].action;
    console.log(`Move ${moveCount + 1}: Action=${bestAction} (${
      ['Up', 'Right', 'Down', 'Left'][bestAction]
    }), Q=${actionScores[0].qValue.toFixed(3)}`);

    return bestAction;
  }

  async function stepAI() {
    if (!aiPlaying) {
      return;
    }

    if (!window.game) {
      console.error('Game not available');
      stopAI();
      return;
    }

    if (isGameOver()) {
      console.log('═══════════════════════════════════════');
      console.log(`Game ${gamesPlayed + 1} finished!`);
      console.log(`Final Score: ${game.score}`);
      console.log(`Max Tile: ${getMaxTile()}`);
      console.log(`Total Moves: ${moveCount}`);
      console.log('═══════════════════════════════════════');

      gamesPlayed++;
      moveCount = 0;

      setTimeout(() => {
        if (aiPlaying) {
          console.log('\nStarting new game...\n');
          game.restart();
          setTimeout(stepAI, AI_DELAY_MS);
        }
      }, 2000);

      return;
    }

    const state = getBoardState();
    if (!state) {
      stopAI();
      return;
    }

    const action = await chooseAction(state);
    game.move(action);
    moveCount++;

    setTimeout(stepAI, AI_DELAY_MS);
  }

  function getMaxTile() {
    if (!window.game || !game.grid) {
      return 0;
    }

    let max = 0;
    game.grid.eachCell(function (x, y, tile) {
      if (tile && tile.value > max) {
        max = tile.value;
      }
    });

    return max;
  }

  window.startAI = async function () {
    if (!window.game) {
      console.error('Game not loaded yet. Please wait and try again.');
      return;
    }

    if (aiPlaying) {
      console.log('AI is already playing');
      return;
    }

    console.log('═══════════════════════════════════════');
    console.log('Starting AI Player');
    console.log('═══════════════════════════════════════');

    const loaded = await loadModel();
    if (!loaded) {
      console.error('Failed to load model. Cannot start AI.');
      return;
    }

    game.restart();
    moveCount = 0;
    aiPlaying = true;

    stepAI();
  };

  window.stopAI = function () {
    if (!aiPlaying) {
      console.log('AI is not playing');
      return;
    }

    aiPlaying = false;
    console.log('\n═══════════════════════════════════════');
    console.log('AI Player stopped');
    console.log('═══════════════════════════════════════');
  };

  document.addEventListener('keydown', function (event) {
    if (event.key === 'a' || event.key === 'A') {
      if (aiPlaying) {
        stopAI();
      } else {
        startAI();
      }
    }
  });

  console.log('AI Player loaded. Press "A" or click "Play AI" to start.');
})();

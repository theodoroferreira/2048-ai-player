/**
 * AI Player for 2048 using ONNX Runtime Web
 * Loads a trained Deep Q-Network model and plays the game automatically
 */

(function () {
  'use strict';

  // Configuration
  const AI_DELAY_MS = 100; // Delay between moves to see animations
  const MODEL_PATH = 'model/model.onnx';

  // State
  let aiPlaying = false;
  let session = null;
  let moveCount = 0;
  let gamesPlayed = 0;

  /**
   * Load the ONNX model
   */
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

  /**
   * Get the current board state as a normalized vector
   * Returns a 16-element array with log2 encoding
   */
  function getBoardState() {
    if (!window.game || !game.grid) {
      console.error('Game not ready');
      return null;
    }

    const size = game.size; // Should be 4
    const state = [];

    // Read board row by row (matching Python implementation)
    for (let x = 0; x < size; x++) {
      for (let y = 0; y < size; y++) {
        const tile = game.grid.cells[x][y];
        if (tile && tile.value > 0) {
          // Log2 encoding: 2->1, 4->2, 8->3, etc.
          state.push(Math.log2(tile.value));
        } else {
          state.push(0);
        }
      }
    }

    return state;
  }

  /**
   * Check if the game is over
   */
  function isGameOver() {
    if (!window.game) {
      return true;
    }
    return game.isGameTerminated();
  }

  /**
   * Check if a move is valid (changes the board)
   */
  function isMoveValid(action) {
    if (!window.game || !game.grid) {
      return false;
    }

    // Create a temporary copy to test the move
    const oldCells = game.grid.cells.map(row => row.slice());

    // Simulate the move
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

    // Check if any tile can move in this direction
    for (let x = 0; x < game.size; x++) {
      for (let y = 0; y < game.size; y++) {
        const tile = game.grid.cells[x][y];
        if (tile) {
          const newX = x + vector.x;
          const newY = y + vector.y;

          // Can move to empty cell
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

  /**
   * Choose the best action using the trained model
   */
  async function chooseAction(state) {
    const sess = await loadModel();

    if (!sess) {
      // Fallback: random valid action if model isn't loaded
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

    // Prepare input tensor [1, 16]
    const inputTensor = new ort.Tensor('float32', new Float32Array(state), [1, 16]);

    // Run inference
    const feeds = {};
    feeds[sess.inputNames[0]] = inputTensor;
    const results = await sess.run(feeds);

    // Get Q-values from output
    const outputTensor = results[sess.outputNames[0]];
    const qValuesArray = outputTensor.data; // Float32Array with 4 Q-values

    // Find valid actions and their Q-values
    const actionScores = [];
    for (let i = 0; i < 4; i++) {
      if (isMoveValid(i)) {
        actionScores.push({ action: i, qValue: qValuesArray[i] });
      }
    }

    // If no valid actions (shouldn't happen), return 0
    if (actionScores.length === 0) {
      console.warn('No valid actions available');
      return 0;
    }

    // Choose action with highest Q-value among valid actions
    actionScores.sort((a, b) => b.qValue - a.qValue);

    const bestAction = actionScores[0].action;
    console.log(`Move ${moveCount + 1}: Action=${bestAction} (${
      ['Up', 'Right', 'Down', 'Left'][bestAction]
    }), Q=${actionScores[0].qValue.toFixed(3)}`);

    return bestAction;
  }

  /**
   * Execute one AI step
   */
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

      // Optionally restart automatically
      setTimeout(() => {
        if (aiPlaying) {
          console.log('\nStarting new game...\n');
          game.restart();
          setTimeout(stepAI, AI_DELAY_MS);
        }
      }, 2000);

      return;
    }

    // Get current state
    const state = getBoardState();
    if (!state) {
      stopAI();
      return;
    }

    // Choose and execute action
    const action = await chooseAction(state);
    game.move(action);
    moveCount++;

    // Schedule next step
    setTimeout(stepAI, AI_DELAY_MS);
  }

  /**
   * Get the maximum tile value on the board
   */
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

  /**
   * Start the AI player
   */
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

    // Load model first
    const loaded = await loadModel();
    if (!loaded) {
      console.error('Failed to load model. Cannot start AI.');
      return;
    }

    // Start a new game
    game.restart();
    moveCount = 0;
    aiPlaying = true;

    // Start playing
    stepAI();
  };

  /**
   * Stop the AI player
   */
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

  // Add keyboard shortcut to toggle AI
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

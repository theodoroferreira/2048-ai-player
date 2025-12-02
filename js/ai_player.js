(function () {
  const AI_DELAY_MS = 100; // delay entre movimentos pra gente conseguir ver a animação
  let aiPlaying = false;
  let model = null; // modelo TensorFlow.js

  async function loadModel() {
    if (model) return model;
    try {
      // Carrega o modelo salvo em /model/model.json
      model = await tf.loadLayersModel('model/model.json');
      console.log('Modelo TensorFlow carregado com sucesso.');
    } catch (err) {
      console.error('Erro ao carregar o modelo TensorFlow:', err);
      model = null;
    }
    return model;
  }

  function getBoardState() {
    if (!window.game || !game.grid) {
      console.error("Game ainda não está pronto.");
      return null;
    }

    const size = game.size; // 4
    const state = [];

    // percorre linha a linha (y depois x)
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const tile = game.grid.cells[x][y];
        if (tile) {
          state.push(Math.log2(tile.value));
        } else {
          state.push(0);
        }
      }
    }
    return state; // vetor de 16 números
  }

  function isGameOver() {
    if (!window.game) return true;
    if (typeof game.isGameTerminated === 'function') {
      return game.isGameTerminated();
    }
    // fallback se for uma versão mais antiga
    return game.over || game.won;
  }

  // Escolhe a ação com base no modelo TF.js
  async function chooseActionFromModel(state) {
    const m = await loadModel();
    if (!m) {
      // fallback: ação aleatória se o modelo não carregou
      return Math.floor(Math.random() * 4);
    }

    // tf.tidy pra evitar vazamento de memória de tensores
    return tf.tidy(() => {
      const input = tf.tensor([state], [1, state.length], 'float32'); // shape [1,16]
      const q = m.predict(input); // esperado shape [1,4]
      const qData = q.dataSync(); // Float32Array com 4 Q-values

      let bestAction = 0;
      for (let i = 1; i < qData.length; i++) {
        if (qData[i] > qData[bestAction]) {
          bestAction = i;
        }
      }
      return bestAction; // 0=cima,1=dir,2=baixo,3=esq
    });
  }

  async function stepAI() {
    if (!aiPlaying) return;

    if (!window.game) {
      console.error("Game não está disponível.");
      aiPlaying = false;
      return;
    }

    if (isGameOver()) {
      console.log("Jogo terminou. Pontuação:", game.score);
      aiPlaying = false;
      return;
    }

    const state = getBoardState();
    if (!state) {
      aiPlaying = false;
      return;
    }

    const action = await chooseActionFromModel(state);
    game.move(action);

    // agenda próximo passo
    setTimeout(() => {
      stepAI(); // não precisa await aqui
    }, AI_DELAY_MS);
  }

  // Tornar funções acessíveis via HTML/console
  window.startAI = function () {
    if (!window.game) {
      console.error("Game ainda não carregou.");
      return;
    }
    aiPlaying = true;
    stepAI();
  };

  window.stopAI = function () {
    aiPlaying = false;
  };
})();

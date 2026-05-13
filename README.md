<div align="center">
  <h1>🧊 Rubik's Cube RL Solver</h1>
  <p>A Java-based Deep Reinforcement Learning Agent that learns to solve a 2x2 Rubik's Cube from scratch using Deep Q-Networks (DQN).</p>
</div>

---

## 🌟 What is this?
This project is a reinforcement learning model based off the deeplearning4j library. It starts off with 0 knowledge of how the cube works. Through trial and error, it slowly learns, getting rewarded for improvement.

### Key Features
- **Mechanics**: Features a mathematically complete 2x2 cube environment with all 12 rotational actions (`U, U', D, D', F, F', B, B', L, L', R, R'`).
- **Reward**: The agent is rewarded for placing tiles on their correct faces.
- **Deeplearning4j**: Able to be ran completely locally on your CPU (or optionally, NVIDIA GPU).

### Improvements to be made
- A punishment system to punish the algorithim for taking too many moves 
- A way to prevent the bug where the algorithim loves to just hang out in an area where it thinks its getting good reward instead of finishing after some point, resulting in plateuing stats

---

## 📊 Live Training Dashboard
As the Java agent trains, it streams its telemetry data. You can open the dashboard by running your local server address:
- **Solve Rate**
- **Epsilon (Exploration Rate)**
- **Live Cube Net**

---

## 🧠 How the AI works (Under the Hood)

1. **State Encoding**: The 2x2 cube has 24 individual colored tiles. We one-hot encode these colors into an array of 144 inputs.
2. **Deep Q-Network (DQN)**: The network has 3 dense hidden layers (with 512 -> 512 -> 256 nodes) using ReLU activation. It takes the 144-element state array and outputs 12 Q-values (representing the expected future reward for taking each of the 12 possible face rotations).
3. **Experience Replay**: The agent stores its past moves in a memory buffer. Every step, it pulls a random batch of 64 past experiences and trains the network on them.
4. **Target Network**: Used to calculate the future rewards, which is synced with the main network every 50 episodes.

## ⚙️ Configuration
Want to tweak the RL module? Open `src/main/java/RubiksCubeRL.java` and modify these constants at the top of the file:
- `MAX_STEPS`: How many moves the agent gets per episode.
- `BATCH_SIZE`: How many experiences to learn from at once.
- `GAMMA`: The discount factor (0.95 means it values future rewards highly).

### GPU Support
Training on a CPU works great, but an NVIDIA GPU is much faster. If you have CUDA 11.6 installed, edit `pom.xml`:
1. Comment out `<artifactId>nd4j-native-platform</artifactId>`
2. Uncomment `<artifactId>nd4j-cuda-11.6-platform</artifactId>`

## Good Luck and Happy Training :))

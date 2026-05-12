<div align="center">
  <h1>🧊 Rubik's Cube RL Solver</h1>
  <p>A Java-based Deep Reinforcement Learning Agent that learns to solve a 2x2 Rubik's Cube from scratch using Deep Q-Networks (DQN).</p>
</div>

---

## 🌟 What is this?
This project demonstrates how a machine learning agent can learn complex sequential puzzle-solving without any hardcoded algorithms or human heuristics. The agent starts knowing absolutely nothing about how a Rubik's Cube works. Through trial and error (Reinforcement Learning), it explores millions of cube states, gets "rewarded" when it matches tiles correctly, and slowly builds a neural network policy to solve the puzzle.

### Key Features
- **Authentic Mechanics**: Features a mathematically complete 2x2 cube environment with all 12 rotational actions (`U, U', D, D', F, F', B, B', L, L', R, R'`).
- **Adaptive Curriculum**: The AI learns like a human! It starts with 1-move scrambles. Once it proves it can solve those consistently (>75% solve rate), the curriculum automatically advances to 2-move scrambles, then 3, all the way to fully scrambled cubes.
- **Reward Shaping**: The agent is rewarded for placing tiles on their correct faces, providing a gradient of "hot/cold" rather than just a binary "solved/not solved" signal.
- **Deeplearning4j**: Powered by the enterprise-grade DL4J library, running completely locally on your CPU (or optionally, NVIDIA GPU).

---

## 📊 Live Training Dashboard
Watching thousands of console logs is boring. We've built a **premium web dashboard** that connects to your local training session in real-time.

As the Java agent trains, it streams telemetry data. You can open the dashboard in your browser to watch the AI learn:
- **Rolling Solve Rate**: Watch the win-rate spike as the agent masters a scramble depth.
- **Epsilon (Exploration Rate)**: See how the agent shifts from random flailing (exploration) to deliberate, learned actions (exploitation).
- **Live 2D Cube Net**: A visual representation of exactly how the cube looks at the current step.

---

## 🚀 Getting Started

### Prerequisites
- **Java 11 or higher** installed (`java -version`).
- **Maven** (Don't have it? Don't worry! Windows users can just use our provided scripts which handle Maven for you).

### 1. Start the Training Agent
The neural network needs to run in the background to solve the cubes.
- **Windows Users**: Simply double-click **`run.bat`** in the project folder. It will auto-detect Java, compile the code, and start training.
- **Mac/Linux/Terminal Users**: 
  ```bash
  mvn compile exec:java
  ```

### 2. Open the Dashboard
Once the agent is running, it will generate a `training_metrics.csv` and `cube_state.json` file.
To view the live dashboard:
- **Windows Users**: Double-click **`serve_dashboard.bat`**. This starts a local web server and gives you a link to click.
- **Terminal Users**: Start a python server in the directory:
  ```bash
  python -m http.server 8000
  ```
  Then navigate to `http://localhost:8000/dashboard.html` in your web browser.

---

## 🧠 How the AI works (Under the Hood)

1. **State Encoding**: The 2x2 cube has 24 individual colored tiles. We one-hot encode these colors into an array of 144 inputs.
2. **Deep Q-Network (DQN)**: The network has 3 dense hidden layers (512 -> 512 -> 256 nodes) using ReLU activation. It takes the 144-element state array and outputs 12 Q-values (representing the expected future reward for taking each of the 12 possible face rotations).
3. **Experience Replay**: The agent stores its past moves in a memory buffer. Every step, it pulls a random batch of 64 past experiences and trains the network on them, preventing catastrophic forgetting.
4. **Target Network**: To stabilize learning, a secondary "Target" network is used to calculate the future rewards, which is slowly synced with the main network every 50 episodes.

## ⚙️ Configuration
Want to tweak the AI? Open `src/main/java/RubiksCubeRL.java` and modify these constants at the top of the file:
- `MAX_STEPS`: How many moves the agent gets per episode.
- `BATCH_SIZE`: How many experiences to learn from at once.
- `GAMMA`: The discount factor (0.95 means it values future rewards highly).

### GPU Support
Training on a CPU works great, but an NVIDIA GPU is much faster. If you have CUDA 11.6 installed, edit `pom.xml`:
1. Comment out `<artifactId>nd4j-native-platform</artifactId>`
2. Uncomment `<artifactId>nd4j-cuda-11.6-platform</artifactId>`

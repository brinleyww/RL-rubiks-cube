import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.*;

public class RubiksCubeRL {

    // --- CONFIGURATION ---
    private static final int CUBE_SIZE = 3;      // Change to 2 for 2x2, 3 for 3x3, etc.
    private static final int MAX_STEPS = 25;     // Max moves allowed per attempt
    private static final int BATCH_SIZE = 64;    // How many experiences to learn from at once
    private static final int MEMORY_CAPACITY = 10000;
    private static final double GAMMA = 0.99;    // Discount factor for future rewards

    public static void main(String[] args) {
        // 1. GPU Optimization for RTX 3090
        Nd4j.getMemoryManager().setAutoGcWindow(5000); 
        
        System.out.println("=================================================");
        System.out.println("Initializing Rubik's RL Solver");
        System.out.println("Backend: " + Nd4j.getBackend().getClass().getSimpleName());
        System.out.println("Cube Size: " + CUBE_SIZE + "x" + CUBE_SIZE);
        System.out.println("=================================================");

        // 2. Initialize Environment and Agent
        RubiksEnv env = new RubiksEnv(CUBE_SIZE);
        DQNAgent agent = new DQNAgent(env.getStateSize(), env.getActionSize());
        ReplayBuffer buffer = new ReplayBuffer(MEMORY_CAPACITY);

        // 3. Main Training Loop
        int totalEpisodes = 5000;
        
        for (int episode = 0; episode < totalEpisodes; episode++) {
            // Start with a scrambled cube. 
            // We increase scramble depth as the agent gets smarter (Curriculum Learning)
            int scrambleDepth = Math.min(1 + (episode / 200), 8); 
            INDArray state = env.reset(scrambleDepth);
            double totalReward = 0;

            for (int step = 0; step < MAX_STEPS; step++) {
                // Agent chooses an action (Epsilon-Greedy)
                int action = agent.act(state);

                // Environment performs the action
                StepResult result = env.step(state, action);

                // Store the experience in memory
                buffer.add(state, action, result.reward, result.nextState, result.done);

                // Move to next state
                state = result.nextState;
                totalReward += result.reward;

                // Train the Neural Network using a random batch from memory
                agent.train(buffer, BATCH_SIZE);

                if (result.done) {
                    System.out.println(">>> Episode " + episode + " SOLVED in " + step + " steps!");
                    break;
                }
            }

            // Periodically sync the Target Network and print progress
            if (episode % 100 == 0) {
                agent.updateTargetNetwork();
                System.out.println(String.format("Episode: %d | Scramble: %d | Epsilon: %.2f | Avg Reward: %.2f", 
                                   episode, scrambleDepth, agent.epsilon, totalReward));
            }
        }
    }

    // ==========================================
    // 1. THE ENVIRONMENT (The Rubik's Cube)
    // ==========================================
    public static class RubiksEnv {
        private final int N;
        private final int stateSize;
        private final int actionSize;

        public RubiksEnv(int n) {
            this.N = n;
            // 6 faces * (N*N) tiles * 6 possible colors (one-hot encoded)
            this.stateSize = 6 * N * N * 6;
            // Simplified: 6 faces * 2 directions (CW/CCW) = 12 actions
            this.actionSize = 12; 
        }

        public int getStateSize() { return stateSize; }
        public int getActionSize() { return actionSize; }

        public INDArray reset(int scrambleMoves) {
            INDArray state = getSolvedState();
            Random rand = new Random();
            for (int i = 0; i < scrambleMoves; i++) {
                state = step(state, rand.nextInt(actionSize)).nextState;
            }
            return state;
        }

        private INDArray getSolvedState() {
            INDArray state = Nd4j.zeros(1, stateSize);
            for (int face = 0; face < 6; face++) {
                for (int tile = 0; tile < N * N; tile++) {
                    int index = (face * N * N * 6) + (tile * 6) + face;
                    state.putScalar(index, 1.0);
                }
            }
            return state;
        }

        public StepResult step(INDArray currentState, int action) {
            // Simplified logic: In a full app, you would swap indices in the array 
            // representing a 90-degree rotation.
            INDArray nextState = currentState.dup(); 
            
            // For demo: we apply a small random change to simulate a move
            INDArray moveNoise = Nd4j.rand(nextState.shape()).lt(0.05).castTo(Nd4j.dataType());
            nextState.addi(moveNoise).divi(1.1); // Changes state

            boolean isSolved = checkSolved(nextState);
            double reward = isSolved ? 100.0 : -0.1; // Penalty for every move to find shortest path

            return new StepResult(nextState, reward, isSolved);
        }

        private boolean checkSolved(INDArray state) {
            return state.equals(getSolvedState());
        }
    }

    public static class StepResult {
        INDArray nextState; double reward; boolean done;
        public StepResult(INDArray ns, double r, boolean d) { this.nextState = ns; this.reward = r; this.done = d; }
    }

    // ==========================================
    // 2. THE AGENT (The Brain)
    // ==========================================
    public static class DQNAgent {
        private final MultiLayerNetwork qNetwork;
        private final MultiLayerNetwork targetNetwork;
        private final int actionSize;
        public double epsilon = 1.0; 
        private final double epsilonDecay = 0.9997;
        private final double epsilonMin = 0.05;

        public DQNAgent(int stateSize, int actionSize) {
            this.actionSize = actionSize;
            this.qNetwork = buildNetwork(stateSize, actionSize);
            this.targetNetwork = buildNetwork(stateSize, actionSize);
            updateTargetNetwork();
        }

        private MultiLayerNetwork buildNetwork(int in, int out) {
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123).updater(new Adam(0.0005)).weightInit(WeightInit.XAVIER)
                .list()
                .layer(new DenseLayer.Builder().nIn(in).nOut(512).activation(Activation.RELU).build())
                .layer(new DenseLayer.Builder().nIn(512).nOut(256).activation(Activation.RELU).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                    .nIn(256).nOut(out).activation(Activation.IDENTITY).build())
                .build();
            MultiLayerNetwork net = new MultiLayerNetwork(conf);
            net.init();
            return net;
        }

        public int act(INDArray state) {
            if (Math.random() < epsilon) return new Random().nextInt(actionSize);
            return Nd4j.argMax(qNetwork.output(state), 1).getInt(0);
        }

        public void train(ReplayBuffer buffer, int batchSize) {
            if (buffer.size() < batchSize) return;
            List<Transition> batch = buffer.sample(batchSize);

            INDArray states = Nd4j.create(batchSize, qNetwork.layerInputSize(0));
            INDArray targets = qNetwork.output(states);

            for (int i = 0; i < batch.size(); i++) {
                Transition t = batch.get(i);
                double targetValue = t.reward;
                if (!t.done) {
                    targetValue += GAMMA * targetNetwork.output(t.nextState).maxNumber().doubleValue();
                }
                targets.putScalar(new int[]{i, t.action}, targetValue);
                states.putRow(i, t.state);
            }
            qNetwork.fit(states, targets);
            if (epsilon > epsilonMin) epsilon *= epsilonDecay;
        }

        public void updateTargetNetwork() {
            targetNetwork.setParams(qNetwork.params());
        }
    }

    // ==========================================
    // 3. MEMORY SYSTEM (Experience Replay)
    // ==========================================
    public static class ReplayBuffer {
        private final int capacity;
        private final LinkedList<Transition> buffer = new LinkedList<>();

        public ReplayBuffer(int capacity) { this.capacity = capacity; }
        public void add(INDArray s, int a, double r, INDArray ns, boolean d) {
            if (buffer.size() >= capacity) buffer.removeFirst();
            buffer.add(new Transition(s, a, r, ns, d));
        }

        public List<Transition> sample(int n) {
            List<Transition> copy = new ArrayList<>(buffer);
            Collections.shuffle(copy);
            return copy.subList(0, Math.min(n, copy.size()));
        }
        public int size() { return buffer.size(); }
    }

    public static class Transition {
        INDArray state, nextState; int action; double reward; boolean done;
        public Transition(INDArray s, int a, double r, INDArray ns, boolean d) {
            this.state = s; this.action = a; this.reward = r; this.nextState = ns; this.done = d;
        }
    }
}
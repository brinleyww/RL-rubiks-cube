import java.io.*;
import java.nio.file.*;
import java.util.*;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class RubiksCubeRL {

    // --- CONFIGURATION ---
    private static final int N = 2;
    private static final int NUM_FACES = 6;
    private static final int NUM_COLORS = 6;
    private static final int STATE_SIZE = NUM_FACES * N * N * NUM_COLORS; 
    private static final int ACTION_SIZE = 12;
    private static final int MAX_STEPS = 25;
    private static final int BATCH_SIZE = 128;
    private static final int MEMORY_CAPACITY = 100000;
    private static final double GAMMA = 0.95;
    private static final int TOTAL_EPISODES = 200000;
    private static final int LOG_INTERVAL = 10;
    private static final int TARGET_UPDATE_INTERVAL = 100;
    
    // MULTI-AGENT SETTINGS
    private static final int NUM_PARALLEL_ENVS = 16; 

    public static void main(String[] args) throws Exception {
        Nd4j.setDataType(DataType.FLOAT);
        Nd4j.getMemoryManager().setAutoGcWindow(5000);

        String baseDir = System.getProperty("user.dir");
        Path metricsPath = Paths.get(baseDir, "training_metrics.csv");
        Path statePath = Paths.get(baseDir, "cube_state.json");

        System.out.println("==========================================================");
        System.out.println("  Rubik's Cube AI v2.0 - Parallel Multi-Agent Training");
        System.out.println("  Backend: " + Nd4j.getBackend().getClass().getSimpleName());
        System.out.println("  Parallel Agents: " + NUM_PARALLEL_ENVS);
        System.out.println("  State size: " + STATE_SIZE + " | Actions: " + ACTION_SIZE);
        System.out.println("==========================================================");

        DQNAgent agent = new DQNAgent(STATE_SIZE, ACTION_SIZE);
        ReplayBuffer buffer = new ReplayBuffer(MEMORY_CAPACITY);
        MetricsLogger logger = new MetricsLogger(metricsPath);

        // Stats tracking
        LinkedList<Boolean> solveHistory = new LinkedList<>();
        LinkedList<Integer> stepsHistory = new LinkedList<>();
        LinkedList<Double> rewardHistory = new LinkedList<>();
        int bestStreak = 0, currentStreak = 0, totalSolves = 0, episodesFinished = 0;
        int scrambleDepth = 3;

        // Init environments
        Environment[] envs = new Environment[NUM_PARALLEL_ENVS];
        for (int i = 0; i < NUM_PARALLEL_ENVS; i++) envs[i] = new Environment(scrambleDepth);

        while (episodesFinished < TOTAL_EPISODES) {
   
            agent.epsilon = Math.max(0.02, Math.pow(0.9999, episodesFinished));

            INDArray batchedStates = Nd4j.create(NUM_PARALLEL_ENVS, STATE_SIZE);
            for (int i = 0; i < NUM_PARALLEL_ENVS; i++) {
                batchedStates.putRow(i, encodeState(envs[i].cube));
            }

            int[] actions = agent.actBatched(batchedStates);

  
            for (int i = 0; i < NUM_PARALLEL_ENVS; i++) {
                Environment e = envs[i];
                int[][] oldCube = copyCube(e.cube);
                INDArray oldState = encodeState(oldCube);
                
                int action = actions[i];
                applyMove(e.cube, action);
                e.steps++;

                double correctBefore = correctTileRatio(oldCube);
                double correctAfter = correctTileRatio(e.cube);
                boolean solved = isSolved(e.cube);
                
                double reward = solved ? 100.0 : ((correctAfter - correctBefore) * 10.0 - 0.1);
                INDArray nextState = encodeState(e.cube);
                
                buffer.add(oldState, action, reward, nextState, solved);
                e.totalReward += reward;

                if (solved || e.steps >= MAX_STEPS) {
              
                    episodesFinished++;
                    solveHistory.add(solved);
                    stepsHistory.add(e.steps);
                    rewardHistory.add(e.totalReward);
                    
                    if (solveHistory.size() > 100) {
                        solveHistory.removeFirst(); stepsHistory.removeFirst(); rewardHistory.removeFirst();
                    }

                    if (solved) {
                        totalSolves++; currentStreak++;
                        bestStreak = Math.max(bestStreak, currentStreak);
                    } else {
                        currentStreak = 0;
                    }

          
                    if (episodesFinished % LOG_INTERVAL == 0) {
                        double solveRate = solveHistory.stream().mapToInt(b -> b ? 1 : 0).average().orElse(0);
                        double avgSteps = stepsHistory.stream().mapToInt(Integer::intValue).average().orElse(0);
                        double avgReward = rewardHistory.stream().mapToDouble(Double::doubleValue).average().orElse(0);
                        
                        logger.log(episodesFinished, scrambleDepth, solved, e.steps, e.totalReward,
                                   agent.epsilon, solveRate, avgSteps, avgReward,
                                   totalSolves, bestStreak, correctTileRatio(e.cube));
                        
                        writeCubeState(statePath, e.cube, episodesFinished, solved, e.steps,
                                       solveRate, avgReward, agent.epsilon, scrambleDepth, totalSolves, bestStreak);
                        
                        if (episodesFinished % 100 == 0) {
                             System.out.printf("Ep %6d | Scr: %d | Win%%: %.1f%% | AvgSteps: %.1f | Eps: %.3f | Solves: %d%n",
                                episodesFinished, scrambleDepth, solveRate * 100, avgSteps, agent.epsilon, totalSolves);
                        }
                        
                        // Adaptive curriculum
                        if (episodesFinished % 200 == 0 && solveRate > 0.40 && scrambleDepth < 15) {
                            scrambleDepth++;
                            System.out.println(">>> CURRICULUM UP! Scramble depth: " + scrambleDepth);
                        }
                    }
                    e.reset(scrambleDepth);
                }
            }

            agent.train(buffer, BATCH_SIZE);

            if (episodesFinished % TARGET_UPDATE_INTERVAL == 0) {
                agent.updateTargetNetwork();
            }
        }
    }

    static class Environment {
        int[][] cube;
        int steps;
        double totalReward;
        Environment(int scramble) { reset(scramble); }
        void reset(int scramble) {
            cube = solvedCube();
            scramble(cube, scramble);
            steps = 0;
            totalReward = 0;
        }
    }

    // CUBE
    static int[][] solvedCube() {
        int[][] cube = new int[6][4];
        for (int f = 0; f < 6; f++) Arrays.fill(cube[f], f);
        return cube;
    }
    static int[][] copyCube(int[][] cube) {
        int[][] c = new int[6][];
        for (int f = 0; f < 6; f++) c[f] = cube[f].clone();
        return c;
    }
    static boolean isSolved(int[][] cube) {
        for (int f = 0; f < 6; f++) {
            for (int t = 1; t < 4; t++) if (cube[f][t] != cube[f][0]) return false;
        }
        return true;
    }
    static double correctTileRatio(int[][] cube) {
        int correct = 0;
        for (int f = 0; f < 6; f++) {
            for (int t = 0; t < 4; t++) if (cube[f][t] == f) correct++;
        }
        return (double) correct / 24.0;
    }
    static void scramble(int[][] cube, int moves) {
        Random r = new Random();
        for (int i = 0; i < moves; i++) applyMove(cube, r.nextInt(12));
    }
    static void applyMove(int[][] cube, int action) {
        int face = action / 2;
        boolean cw = (action % 2 == 0);
        int[] f = cube[face];
        if (cw) { int tmp = f[0]; f[0] = f[2]; f[2] = f[3]; f[3] = f[1]; f[1] = tmp; }
        else { int tmp = f[0]; f[0] = f[1]; f[1] = f[3]; f[3] = f[2]; f[2] = tmp; }

        switch (face) {
            case 0: cycleEdges(cube, new int[]{2, 4, 3, 5}, new int[]{0,1}, new int[]{0,1}, new int[]{0,1}, new int[]{0,1}, cw); break;
            case 1: cycleEdges(cube, new int[]{2, 5, 3, 4}, new int[]{2,3}, new int[]{2,3}, new int[]{2,3}, new int[]{2,3}, cw); break;
            case 2: // Front
                if (cw) {
                    int[] t = {cube[0][2], cube[0][3]};
                    cube[0][2] = cube[4][3]; cube[0][3] = cube[4][1];
                    cube[4][1] = cube[1][1]; cube[4][3] = cube[1][0];
                    cube[1][0] = cube[5][2]; cube[1][1] = cube[5][0];
                    cube[5][0] = t[0]; cube[5][2] = t[1];
                } else {
                    int[] t = {cube[0][2], cube[0][3]};
                    cube[0][2] = cube[5][0]; cube[0][3] = cube[5][2];
                    cube[5][0] = cube[1][1]; cube[5][2] = cube[1][0];
                    cube[1][0] = cube[4][3]; cube[1][1] = cube[4][1];
                    cube[4][1] = t[1]; cube[4][3] = t[0];
                }
                break;
            case 3: // Back
                if (cw) {
                    int[] t = {cube[0][0], cube[0][1]};
                    cube[0][0] = cube[5][1]; cube[0][1] = cube[5][3];
                    cube[5][1] = cube[1][3]; cube[5][3] = cube[1][2];
                    cube[1][2] = cube[4][0]; cube[1][3] = cube[4][2];
                    cube[4][0] = t[1]; cube[4][2] = t[0];
                } else {
                    int[] t = {cube[0][0], cube[0][1]};
                    cube[0][0] = cube[4][2]; cube[0][1] = cube[4][0];
                    cube[4][0] = cube[1][3]; cube[4][2] = cube[1][2];
                    cube[1][2] = cube[5][3]; cube[1][3] = cube[5][1];
                    cube[5][1] = t[0]; cube[5][3] = t[1];
                }
                break;
            case 4: // Left
                if (cw) {
                    int[] t = {cube[0][0], cube[0][2]};
                    cube[0][0] = cube[3][3]; cube[0][2] = cube[3][1];
                    cube[3][1] = cube[1][2]; cube[3][3] = cube[1][0];
                    cube[1][0] = cube[2][0]; cube[1][2] = cube[2][2];
                    cube[2][0] = t[0]; cube[2][2] = t[1];
                } else {
                    int[] t = {cube[0][0], cube[0][2]};
                    cube[0][0] = cube[2][0]; cube[0][2] = cube[2][2];
                    cube[2][0] = cube[1][0]; cube[2][2] = cube[1][2];
                    cube[1][0] = cube[3][3]; cube[1][2] = cube[3][1];
                    cube[3][1] = t[1]; cube[3][3] = t[0];
                }
                break;
            case 5: // Right
                if (cw) {
                    int[] t = {cube[0][1], cube[0][3]};
                    cube[0][1] = cube[2][1]; cube[0][3] = cube[2][3];
                    cube[2][1] = cube[1][1]; cube[2][3] = cube[1][3];
                    cube[1][1] = cube[3][2]; cube[1][3] = cube[3][0];
                    cube[3][0] = t[1]; cube[3][2] = t[0];
                } else {
                    int[] t = {cube[0][1], cube[0][3]};
                    cube[0][1] = cube[3][2]; cube[0][3] = cube[3][0];
                    cube[3][0] = cube[1][3]; cube[3][2] = cube[1][1];
                    cube[1][1] = cube[2][1]; cube[1][3] = cube[2][3];
                    cube[2][1] = t[0]; cube[2][3] = t[1];
                }
                break;
        }
    }
    static void cycleEdges(int[][] cube, int[] f, int[] i0, int[] i1, int[] i2, int[] i3, boolean cw) {
        int[][] idx = {i0, i1, i2, i3};
        if (cw) {
            int[] t = {cube[f[3]][idx[3][0]], cube[f[3]][idx[3][1]]};
            for (int i = 3; i > 0; i--) {
                cube[f[i]][idx[i][0]] = cube[f[i-1]][idx[i-1][0]];
                cube[f[i]][idx[i][1]] = cube[f[i-1]][idx[i-1][1]];
            }
            cube[f[0]][idx[0][0]] = t[0]; cube[f[0]][idx[0][1]] = t[1];
        } else {
            int[] t = {cube[f[0]][idx[0][0]], cube[f[0]][idx[0][1]]};
            for (int i = 0; i < 3; i++) {
                cube[f[i]][idx[i][0]] = cube[f[i+1]][idx[i+1][0]];
                cube[f[i]][idx[i][1]] = cube[f[i+1]][idx[i+1][1]];
            }
            cube[f[3]][idx[3][0]] = t[0]; cube[f[3]][idx[3][1]] = t[1];
        }
    }
    static INDArray encodeState(int[][] cube) {
        INDArray state = Nd4j.zeros(1, STATE_SIZE);
        for (int f = 0; f < 6; f++) {
            for (int t = 0; t < 4; t++) {
                int idx = (f * 24) + (t * 6) + cube[f][t];
                state.putScalar(new int[]{0, idx}, 1.0f);
            }
        }
        return state;
    }

    public static class DQNAgent {
        private final MultiLayerNetwork qNetwork, targetNetwork;
        public double epsilon = 1.0;
        public DQNAgent(int in, int out) {
            this.qNetwork = build(in, out); this.targetNetwork = build(in, out);
            targetNetwork.setParams(qNetwork.params());
        }
        private MultiLayerNetwork build(int in, int out) {
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123).updater(new Adam(0.0005)).weightInit(WeightInit.XAVIER)
                .list()
                .layer(new DenseLayer.Builder().nIn(in).nOut(512).activation(Activation.RELU).build())
                .layer(new DenseLayer.Builder().nIn(512).nOut(512).activation(Activation.RELU).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE).nIn(512).nOut(out).activation(Activation.IDENTITY).build())
                .build();
            MultiLayerNetwork net = new MultiLayerNetwork(conf); net.init(); return net;
        }
        public int[] actBatched(INDArray states) {
            int n = (int) states.size(0);
            int[] actions = new int[n];
            INDArray output = qNetwork.output(states);
            Random rand = new Random();
            for (int i = 0; i < n; i++) {
                if (Math.random() < epsilon) actions[i] = rand.nextInt(12);
                else actions[i] = Nd4j.argMax(output.getRow(i)).getInt(0);
            }
            return actions;
        }
        public void train(ReplayBuffer buffer, int batchSize) {
            if (buffer.size() < batchSize) return;
            List<Transition> batch = buffer.sample(batchSize);
            INDArray states = Nd4j.create(batchSize, STATE_SIZE);
            INDArray nextStates = Nd4j.create(batchSize, STATE_SIZE);
            for (int i = 0; i < batchSize; i++) {
                states.putRow(i, batch.get(i).state);
                nextStates.putRow(i, batch.get(i).nextState);
            }
            INDArray targets = qNetwork.output(states);
            INDArray nextQ = targetNetwork.output(nextStates);
            for (int i = 0; i < batchSize; i++) {
                Transition t = batch.get(i);
                double val = t.reward + (t.done ? 0 : GAMMA * nextQ.getRow(i).maxNumber().doubleValue());
                targets.putScalar(new int[]{i, t.action}, (float)val);
            }
            qNetwork.fit(states, targets);
        }
        public void updateTargetNetwork() { targetNetwork.setParams(qNetwork.params()); }
    }

    public static class ReplayBuffer {
        private final int cap;
        private final LinkedList<Transition> buffer = new LinkedList<>();
        public ReplayBuffer(int c) { this.cap = c; }
        public void add(INDArray s, int a, double r, INDArray ns, boolean d) {
            if (buffer.size() >= cap) buffer.removeFirst();
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

    public static class MetricsLogger {
        private final Path p;
        public MetricsLogger(Path p) throws Exception {
            this.p = p;
            try (PrintWriter w = new PrintWriter(new BufferedWriter(new FileWriter(p.toFile(), false)))) {
                w.println("episode,scramble,solved,steps,reward,epsilon,solve_rate,avg_steps,avg_reward,total_solves,best_streak,tile_accuracy");
            }
        }
        public void log(int e, int s, boolean sl, int st, double r, double ep, double sr, double as, double ar, int ts, int bs, double ta) throws Exception {
            try (PrintWriter w = new PrintWriter(new BufferedWriter(new FileWriter(p.toFile(), true)))) {
                w.printf("%d,%d,%s,%d,%.4f,%.6f,%.4f,%.2f,%.4f,%d,%d,%.4f%n", e, s, sl, st, r, ep, sr, as, ar, ts, bs, ta);
            }
        }
        public void close() {}
    }

    static void writeCubeState(Path p, int[][] cube, int ep, boolean sl, int st, double sr, double ar, double eps, int scr, int ts, int bs) throws Exception {
        StringBuilder sb = new StringBuilder();
        sb.append("{").append("\"episode\":").append(ep).append(",").append("\"solved\":").append(sl).append(",")
          .append("\"steps\":").append(st).append(",").append("\"solveRate\":").append(String.format("%.4f", sr)).append(",")
          .append("\"avgReward\":").append(String.format("%.4f", ar)).append(",").append("\"epsilon\":").append(String.format("%.6f", eps)).append(",")
          .append("\"scramble\":").append(scr).append(",").append("\"totalSolves\":").append(ts).append(",").append("\"bestStreak\":").append(bs).append(",")
          .append("\"cube\":[");
        for (int f = 0; f < 6; f++) {
            sb.append("["); for (int t = 0; t < 4; t++) { sb.append(cube[f][t]); if (t < 3) sb.append(","); } sb.append("]");
            if (f < 5) sb.append(",");
        }
        sb.append("]}");
        Files.writeString(p, sb.toString());
    }
}

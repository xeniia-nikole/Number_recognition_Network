package org.example;

import org.jblas.DoubleMatrix;
import org.jblas.util.Random;
import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;

import java.io.IOException;
import java.io.Serial;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Network implements Serializable {
    @Serial
    private static final long serialVersionUID = 1L;

    /**
     * Rate of precision that indicates when the net should stop learning. Possible
     * values are in range 0..1
     */
    private static final double PARAM_PRECISION_RATE = 0.97;
    private final int layersNum;
    private final DoubleMatrix[] weights;
    private final DoubleMatrix[] biases;

    public Network(int @NotNull ... sizes) {
        this.layersNum = sizes.length;

        this.biases = new DoubleMatrix[sizes.length - 1];
        this.weights = new DoubleMatrix[sizes.length - 1];

        // Storing biases
        for (int i = 1; i < sizes.length; i++) {
            double[][] doubles = new double[sizes[i]][];
            for (int j = 0; j < sizes[i]; j++) {
                double[] temp = new double[]{Random.nextGaussian()};
                doubles[j] = temp;
            }
            biases[i - 1] = new DoubleMatrix(doubles);
        }
        // Storing weights
        for (int i = 1; i < sizes.length; i++) {
            double[][] doubles = new double[sizes[i]][];
            for (int j = 0; j < sizes[i]; j++) {
                double[] temp = new double[sizes[i - 1]];
                for (int k = 0; k < sizes[i - 1]; k++) {
                    temp[k] = Random.nextGaussian();
                }
                doubles[j] = temp;
            }
            weights[i - 1] = new DoubleMatrix(doubles);
        }
    }

    /**
     * @param trainingData  - list of arrays (x, y) representing the training inputs
     *                      and corresponding desired outputs
     * @param epochs        - the number of epochs to train for
     * @param miniBatchSize - the size of the mini-batches to use when sampling
     * @param eta           - the learning rate, η
     * @param testData      - the test data use to evaluate the net
     */
    public void SGD(@NotNull List<double[][]> trainingData, int epochs, int miniBatchSize, double eta,
                    List<double[][]> testData) {

        int testsCounter = 0;

        int size = trainingData.size();

        if (testData != null) {
            testsCounter = testData.size();
        }

        for (int j = 0; j < epochs; j++) {
            Collections.shuffle(trainingData);
            List<List<double[][]>> miniBatches = new ArrayList<>();
            for (int k = 0; k < size; k += miniBatchSize) {
                miniBatches.add(trainingData.subList(k, k + miniBatchSize));
            }
            for (List<double[][]> miniBatch : miniBatches) {
                updateMiniBatch(miniBatch, eta);
            }

            if (testData != null) {
                int e = evaluate(testData);
                System.out.printf("Epoch %d: %d / %d%n", j, e, testsCounter);
                if (e >= testsCounter * PARAM_PRECISION_RATE) {
                    try {
                        Util.serialize(this);
                    } catch (IOException e1) {
                        e1.printStackTrace();
                    }
                    break;
                }
            } else {
                System.out.printf("Epoch %d complete%n", j);
            }
        }
    }

    /**
     * @param testData - the test data used to evaluate the net
     * @return the number of test inputs for which the neural network outputs the
     * correct result
     */
    private int evaluate(@NotNull List<double[][]> testData) {
        int sum = 0;
        for (double[][] inputOutput : testData) {
            DoubleMatrix x = new DoubleMatrix(inputOutput[0]);
            DoubleMatrix y = new DoubleMatrix(inputOutput[1]);
            DoubleMatrix netOutput = feedForward(x);
            if (netOutput.argmax() == y.argmax()) {
                sum++;
            }
        }
        return sum;
    }

    /**
     * @param doubleMatrix - activation vector - the 1st layer also called the input layer
     * @return DoubleMatrix - vector containing output from the network consisting
     * of float numbers between 0 and 1
     */
    public DoubleMatrix feedForward(DoubleMatrix doubleMatrix) {
        for (int i = 0; i < layersNum - 1; i++) {
            double[] doubles = new double[weights[i].rows];
            for (int j = 0; j < weights[i].rows; j++) {
                doubles[j] = weights[i].getRow(j).dot(doubleMatrix) + biases[i].get(j);
            }
            DoubleMatrix output = new DoubleMatrix(doubles);
            doubleMatrix = sigmoid(output);
        }
        return doubleMatrix;
    }

    /**
     * Update the network’s weights and biases by applying gradient descent using
     * backpropagation to a single mini batch. The "mini_batch" is a list of arrays
     * "(x, y)", and "eta" is the learning rate.
     *
     * @param miniBatch - part of a training data
     * @param eta       - the learning rate
     */
    private void updateMiniBatch(List<double[][]> miniBatch, double eta) {
        DoubleMatrix[] nablaB = new DoubleMatrix[biases.length];
        DoubleMatrix[] nablaW = new DoubleMatrix[weights.length];

        for (int i = 0; i < nablaB.length; i++) {
            nablaB[i] = new DoubleMatrix(biases[i].getRows(), biases[i].getColumns());
        }
        for (int i = 0; i < nablaW.length; i++) {
            nablaW[i] = new DoubleMatrix(weights[i].getRows(), weights[i].getColumns());
        }

        for (double[][] inputOutput : miniBatch) {
            DoubleMatrix[][] deltas = backProp(inputOutput);

            DoubleMatrix[] deltaNablaB = deltas[0];
            DoubleMatrix[] deltaNablaW = deltas[1];

            for (int i = 0; i < nablaB.length; i++) {
                nablaB[i] = nablaB[i].add(deltaNablaB[i]);
            }
            for (int i = 0; i < nablaW.length; i++) {
                nablaW[i] = nablaW[i].add(deltaNablaW[i]);
            }
        }
        for (int i = 0; i < biases.length; i++) {
            biases[i] = biases[i].sub(nablaB[i].mul(eta / miniBatch.size()));
        }
        for (int i = 0; i < weights.length; i++) {
            weights[i] = weights[i].sub(nablaW[i].mul(eta / miniBatch.size()));
        }
    }

    /**
     * Return an array (nablaB , nablaW) representing the gradient for the cost
     * function C. "nablaB" and "nablaW" are layer-by-layer arrays of DoubleMatrices
     * , similar to this. biases and this.weights.
     *
     * @param inputsOutputs is getting from method updateMiniBatch
     * @return DoubleMatrix[][]
     */
    @Contract("_ -> new")
    private DoubleMatrix[][] backProp(double[][] inputsOutputs) {
        DoubleMatrix[] nablaB = new DoubleMatrix[biases.length];
        DoubleMatrix[] nablaW = new DoubleMatrix[weights.length];

        for (int i = 0; i < nablaB.length; i++) {
            nablaB[i] = new DoubleMatrix(biases[i].getRows(), biases[i].getColumns());
        }
        for (int i = 0; i < nablaW.length; i++) {
            nablaW[i] = new DoubleMatrix(weights[i].getRows(), weights[i].getColumns());
        }

        // FeedForward
        DoubleMatrix activation = new DoubleMatrix(inputsOutputs[0]);
        DoubleMatrix[] activations = new DoubleMatrix[layersNum];
        activations[0] = activation;
        DoubleMatrix[] matrices = new DoubleMatrix[layersNum - 1];

        for (int i = 0; i < layersNum - 1; i++) {
            double[] scalars = new double[weights[i].rows];
            for (int j = 0; j < weights[i].rows; j++) {
                scalars[j] = weights[i].getRow(j).dot(activation) + biases[i].get(j);
            }
            DoubleMatrix matrix = new DoubleMatrix(scalars);
            matrices[i] = matrix;
            activation = sigmoid(matrix);
            activations[i + 1] = activation;
        }

        // Backward pass
        DoubleMatrix output = new DoubleMatrix(inputsOutputs[1]);
        DoubleMatrix delta = costDerivative(activations[activations.length - 1], output)
                .mul(sigmoidPrime(matrices[matrices.length - 1])); // BP1
        nablaB[nablaB.length - 1] = delta; // BP3
        nablaW[nablaW.length - 1] = delta.mmul(activations[activations.length - 2].transpose()); // BP4
        for (int layer = 2; layer < layersNum; layer++) {
            DoubleMatrix z = matrices[matrices.length - layer];
            DoubleMatrix sp = sigmoidPrime(z);
            delta = weights[weights.length + 1 - layer].transpose().mmul(delta).mul(sp); // BP2
            nablaB[nablaB.length - layer] = delta; // BP3
            nablaW[nablaW.length - layer] = delta.mmul(activations[activations.length - 1 - layer].transpose()); // BP4
        }
        return new DoubleMatrix[][]{nablaB, nablaW};
    }

    /**
     * @param z - input vector created by finding dot product of weights and inputs
     *          and added a bias of a neuron
     * @return output vector - inputs for the next layer
     */
    @Contract("_ -> new")
    private @NotNull DoubleMatrix sigmoid(@NotNull DoubleMatrix z) {
        double[] output = new double[z.length];
        for (int i = 0; i < output.length; i++) {
            output[i] = 1 / (1 + Math.exp(-z.get(i)));
        }
        return new DoubleMatrix(output);
    }

    private DoubleMatrix sigmoidPrime(DoubleMatrix doubleMatrix) {
        return sigmoid(doubleMatrix).mul(sigmoid(doubleMatrix).rsub(1));
    }


    private DoubleMatrix costDerivative(@NotNull DoubleMatrix outputActivations, DoubleMatrix output) {
        return outputActivations.sub(output);
    }

    public void testing(Network deserializeNetwork, @NotNull List<double[][]> testData, int elementIndex) {
        Util util = new Util();
        util.testing(deserializeNetwork ,testData, elementIndex);
    }

    @Override
    public String toString() {
        return "Network main data:\n" +
                "Network's serial number = " + serialVersionUID +
                "\nLayers number = " + layersNum;
    }


}
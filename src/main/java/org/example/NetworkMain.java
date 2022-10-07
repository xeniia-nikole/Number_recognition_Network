package org.example;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Stream;

import mnist.MnistDataReader;
import mnist.MnistMatrix;
import org.jetbrains.annotations.NotNull;

import static org.example.Util.deserialize;

public class NetworkMain {

    private static final String FILE_DATA_10K = "data/t10k-images.idx3-ubyte";
    private static final String FILE_LABELS_10K = "data/t10k-labels.idx1-ubyte";

    private static @NotNull List<double[][]> getTrainingDataFromMnist() throws IOException {
        List<double[][]> trainingData = new ArrayList<>();

        MnistMatrix[] mnistMatrix = new MnistDataReader().readData(FILE_DATA_10K, FILE_LABELS_10K);

        for (MnistMatrix matrix : mnistMatrix) {
            double[][] io = new double[2][];
            double[] x = new double[784];

            for (int r = 0; r < matrix.getNumberOfRows(); r++) {
                for (int c = 0; c < matrix.getNumberOfColumns(); c++) {
                    x[r * matrix.getNumberOfColumns() + c] = (double) matrix.getValue(r, c) / 255;
                }
            }
            double[] y = Stream.iterate(0, d -> d).limit(10).mapToDouble(d -> d).toArray();
            y[matrix.getLabel()] = 1;
            io[0] = x;
            io[1] = y;
            trainingData.add(io);
        }

        return trainingData;
    }

    public static void main(String[] args) throws IOException, ClassNotFoundException {
        trainingAndSerialization();;
        deserializationAndTesting();
    }

    private static void trainingAndSerialization() throws IOException {
        List<double[][]> trainingData = getTrainingDataFromMnist();
        Network network = new Network(784, 30, 10);
        network.SGD(trainingData, 100, 10, 13.0, trainingData);
    }

    private static void deserializationAndTesting() throws IOException, ClassNotFoundException {
        Network deserializeNetwork = deserialize();
        System.out.println("Deserialization was successfully finished\nHello, User");
        System.out.println(deserializeNetwork.toString());
        List<double[][]> testingData = getTrainingDataFromMnist();
        int testingElementIndex = (int) (Math.random() * 9999);
        ;
        deserializeNetwork.
                testing(deserializeNetwork, testingData, testingElementIndex);
    }



}

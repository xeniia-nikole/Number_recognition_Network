package org.example;

import org.jblas.DoubleMatrix;
import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;

import java.io.*;
import java.util.List;

/**
 * Util class for serialization and other stuff. Not the best way to organize
 * methods but code conventions are discarded here
 */
public class Util {

    Util() {

    }

    private static final String FILE_SERIALIZATION = "num_recognition_net.ser";

    /**
     * Serializes object
     *
     * @param obj object to serialize
     * @throws IOException if anything strange with io occurred
     */
    public static void serialize(Object obj) throws IOException {
        FileOutputStream fileOutputStream = new FileOutputStream(FILE_SERIALIZATION);
        try (ObjectOutputStream objectOutputStream = new ObjectOutputStream(fileOutputStream)) {
            objectOutputStream.writeObject(obj);
            objectOutputStream.flush();
        }
        System.out.println("Serialized");
    }

    /**
     * Deserializes to object
     *
     * @return deserialized object
     * @throws IOException            if anything strange with io occurred
     * @throws ClassNotFoundException if class wasn't found :(
     */
    public static Network deserialize() throws IOException, ClassNotFoundException {
        Network network;
        FileInputStream fileInputStream = new FileInputStream(FILE_SERIALIZATION);
        try (ObjectInputStream objectInputStream = new ObjectInputStream(fileInputStream)) {
            Object obj = objectInputStream.readObject();
            network = (Network) obj;
        }
        return network;
    }


    public void testing(Network deserializeNetwork, @NotNull List<double[][]> testData, int elementIndex) {
        int size = testData.size();
        if (elementIndex > size) {
            System.out.println("elementIndex > testData.size()");
        }

        double[][] intInfo = testData.get(elementIndex);
        DoubleMatrix picture = new DoubleMatrix(intInfo[0]);
        DoubleMatrix label = new DoubleMatrix(intInfo[1]);
        System.out.println("Expected number : " + doubleMatrixToString(label));

        DoubleMatrix resultMatrix = deserializeNetwork.feedForward(picture);

        System.out.println("Result : " + doubleMatrixToString(resultMatrix));


    }

    /**
     * Converts DoubleMatrix to String
     *
     * @param dm DoubleMatrix
     * @return String representation of DoubleMatrix
     */
    public static @NotNull String doubleMatrixToString(@NotNull DoubleMatrix dm) {
        StringBuilder sb = new StringBuilder();
        for (double d : dm.toArray()) {
            sb.append(d >= 0.5 ? 1 : 0).append(' ');
        }
        return sb.toString();
    }


    /**
     * Needs to be DONE
     * Prints matrix
     *
     * @param matrix is @NotNull and needs to be DoubleMatrix
     */
    public static void printDoubleMatrix(@NotNull DoubleMatrix matrix) {
        // TO DO
//        System.out.print("[");
//        String str = doubleMatrixToString(matrix);
//        String [] rows = str.split("", 28);
//        for (String s:rows) {
//            System.out.println(s);
//        }
//        System.out.println("]");
    }

    @Contract(pure = true)
    private static int format(@NotNull String s) {
        // TO DO
        String str = s.trim();
        int number = 999;
        String ex0 = "1 0 0 0 0 0 0 0 0 0";
        String ex1 = "0 1 0 0 0 0 0 0 0 0";
        String ex2 = "0 0 1 0 0 0 0 0 0 0";
        String ex3 = "0 0 0 1 0 0 0 0 0 0";
        String ex4 = "0 0 0 0 1 0 0 0 0 0";
        String ex5 = "0 0 0 0 0 1 0 0 0 0";
        String ex6 = "0 0 0 0 0 0 1 0 0 0";
        String ex7 = "0 0 0 0 0 0 0 1 0 0";
        String ex8 = "0 0 0 0 0 0 0 0 1 0";
        String ex9 = "0 0 0 0 0 0 0 0 0 1";

        if (str.equals(ex0)) {
            number = 0;
        } else if (str.equals(ex1)) {
            number = 0;
        } else if (str.equals(ex2)) {
            number = 0;
        } else if (str.equals(ex3)) {
            number = 0;
        } else if (str.equals(ex4)) {
            number = 0;
        } else if (str.equals(ex5)) {
            number = 0;
        } else if (str.equals(ex6)) {
            number = 0;
        } else if (str.equals(ex7)) {
            number = 0;
        } else if (str.equals(ex8)) {
            number = 0;
        } else if (str.equals(ex9)) {
            number = 0;
        }
        return number;
    }
}

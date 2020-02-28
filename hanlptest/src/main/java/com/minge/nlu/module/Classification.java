package com.minge.nlu.module;

import com.minge.nlu.entity.*;

import org.apache.commons.lang3.StringUtils;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import javax.annotation.PostConstruct;
import java.io.Serializable;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.io.Serializable;


public class Classification {
    private List<String> cid2Labels;
    private List<List<String>> labelsList = new ArrayList<>();

    // Iuput node names
    protected final String SEQUENCE_LENGTH_NODE = "model/sequence_length";
    protected final String ENCODER_NODE_1 = "model/encoder_inputs"; // words
    protected final String ENCODER_NODE_2 = "model/encoder_inputs2"; // properties

    // Output node names
    private final String JOINT_PROBABILITIES_NODE = "model/joint_pattern_probabilities";
    private final String MANUAL_PATTERN_PROBABILITIES_NODE = "model/manual_pattern_probabilities";
    private final String QA_PATTERN_PROBABILITIES_NODE = "model/qa_pattern_probabilities";
    private final String CID2_PATTERN_PROBABILITIES_NODE = "model/cid2_pattern_probabilities";
    private final String OTHER_PATTERN_PROBABILITIES_NODE = "model/other_pattern_probabilities";
    private final String KG_PATTERN_PROBABILITIES_NODE = "model/kg_pattern_probabilities";

    protected final int PROPERTIES_SIZE = 6;
    protected final int BUCKET_2_MAX_SIZE = 50;

    protected final String PROPERTIES_SPLITTER = ":";
    protected Map<String, Integer> vocabMap;
    protected Map<String, Integer> propVocabMap;
    protected Session session;

    private int getIndexInVocab(String word) {
        final int DEFAULT_INDEX = 1;
        Integer ret = null;
        if (StringUtils.isNotBlank(word)) {
            ret = vocabMap.get(word);
        }
        return ret != null ? ret : DEFAULT_INDEX;
    }

    private Tensor generateInputFeedWords(List<String> words, int size) {
        int[][] content = new int[1][size];
        for (int i = 0; i < size; i += 1) {
            content[0][i] = getIndexInVocab(words.get(i).replaceAll("[0-9]", "0"));
        }
        return Tensor.create(content);
    }

    private Map<String, Tensor> generateAllInputFeed(List<String> words, List<List<String>> properties) {
        Map<String, Tensor> feed = new HashMap<>();
        int size = words.size();
        feed.put(SEQUENCE_LENGTH_NODE, Tensor.create(new int[]{words.size()}));
        feed.put(ENCODER_NODE_1, generateInputFeedWords(words, size));
        feed.put(ENCODER_NODE_2, generateInputFeedProperties(properties, size));
        return feed;
    }

    private int getIndexInPropVocab(String word) {
        final int DEFAULT_INDEX = 1;
        Integer ret = null;
        if (StringUtils.isNotBlank(word)) {
            ret = propVocabMap.get(word);
        }
        return ret != null ? ret : DEFAULT_INDEX;
    }


    protected int getMaxIndex(float[] arr) {
        return getMaxIndex(arr, 0);
    }

    protected int getMaxIndex(float[] arr, int fromIdx) {
        if (arr == null || fromIdx < 0 || fromIdx > arr.length - 1) {
            throw new IllegalArgumentException();
        }

        float max = Float.MIN_VALUE;
        int maxIndex = fromIdx;
        for (int i = fromIdx; i < arr.length; i += 1) {
            if (arr[i] > max) {
                max = arr[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    private Tensor generateInputFeedProperties(List<List<String>> properties, int size) {
        int[][][] content = new int[1][size][PROPERTIES_SIZE];
        for (int i = 0; i < size; i += 1) {
            List<Integer> indices = new ArrayList<>();
            for (String property : properties.get(i)) {
                // Should at least has 1 property
                indices.add(getIndexInPropVocab(property));
            }
            int index = 0;
            for (int j = 0; j < PROPERTIES_SIZE; j += 1) {
                content[0][i][j] = indices.get(index);
                index += 1;
                if (index == indices.size()) {
                    index = 0;
                }
            }
        }
        return Tensor.create(content);
    }

    protected ModelResponse generateResponse(List<Tensor<?>> tensors, List<String> words) {
        ModelResponse ret = new ModelResponse();
        ret.classificationProbabilities = new ArrayList<>();
        ret.classifications = new ArrayList<>();
        // Probabilities
        for (int i = 0; i < 5; i++) {
            Tensor<?> tensor = tensors.get(i);
            List<String> labels = labelsList.get(i);
            if (tensor.numDimensions() != 2 || tensor.shape()[0] != 1) {
                throw new RuntimeException(String.format("Expected model to produce a [1 N] shaped tensor where N is the number of probabilities, instead it produced one with shape %s", Arrays.toString(tensor.shape())));
            }
            float[] probabilities = tensor.copyTo(new float[1][(int) tensor.shape()[1]])[0];


            ret.classificationProbabilities.add(StringUtils.join(probabilities, '|'));


            try {
                ret.classifications.add(Integer.parseInt(labels.get(getMaxIndex(probabilities))));
            } catch (NumberFormatException e) {
                //logger.warn("Parse intent to integer fail: {}", labels.get(getMaxIndex(probabilities)));
                System.out.println("Parse intent to integer fail: " + labels.get(getMaxIndex(probabilities)));
            }
        }

        Tensor<?> tensor = tensors.get(5);
        if (tensor.numDimensions() != 2 || tensor.shape()[0] != 1) {
            throw new RuntimeException(String.format("Expected model to produce a [1 N] shaped tensor where N is the number of probabilities, instead it produced one with shape %s", Arrays.toString(tensor.shape())));
        }

        float[] probabilities = tensor.copyTo(new float[1][(int) tensor.shape()[1]])[0];
        ret.cid2 = cid2Labels.get(getMaxIndex(probabilities));

        return ret;
    }

    private ModelResponse nerMultiProperties(List<String> words, List<List<String>> properties) {
        Session.Runner runner = session.runner();

        // Support max 50 words
        if (words.size() > BUCKET_2_MAX_SIZE) {
            words = words.subList(0, BUCKET_2_MAX_SIZE);
        }

        if (properties.size() > BUCKET_2_MAX_SIZE) {
            properties = properties.subList(0, BUCKET_2_MAX_SIZE);
        }

        // Input
        Map<String, Tensor> feed = generateAllInputFeed(words, properties);
        for (Map.Entry<String, Tensor> entry : feed.entrySet()) {
            runner.feed(entry.getKey(), entry.getValue());
        }

        // Output
        List<String> fetchNames = getFetchNames();
        for (String name : fetchNames) {
            runner.fetch(name);
        }
        List<Tensor<?>> results = runner.run();

        if (results.size() != fetchNames.size()) {
            throw new RuntimeException(String.format("Model output size should be %s, but it's %s", fetchNames.size(), results.size()));
        }

        ModelResponse ret = generateResponse(results, words);

        // Close input and output tensors
        for (Tensor tensor : feed.values()) {
            tensor.close();
        }
        for (Tensor tensor : results) {
            tensor.close();
        }
        return ret;
    }


    public ModelResponse ner(List<String> words, List<String> properties) {
        List<List<String>> multiProperties = new ArrayList<>();
        for (String property : properties) {
            multiProperties.add(Arrays.asList(property.split(PROPERTIES_SPLITTER)));//[[1,2,4,5,6], [2,6,6,6,4]]
        }
        return nerMultiProperties(words, multiProperties);// "我": [1,2,4,5,6], "你":[2,6,6,6,4]
    }

    private void warmUp() {
        List<String> words = new ArrayList<>();
        words.add("来首");
        words.add("歌");

        List<String> properties = new ArrayList<>();
        properties.add("<blank>");
        properties.add("<music_artist>");

        ModelResponse ret = ner(words, properties);
        System.out.println("分类结果 = " + ret.classifications);
    }

    public void superInit(String modelPath, String vocabPath, String propVocabPath) throws IOException{
        try {
            List<String> vocab = Files.readAllLines(Paths.get(vocabPath));
            vocabMap = new HashMap<>();
            for (int i = 0; i < vocab.size(); i += 1) {
                vocabMap.put(vocab.get(i), i);
            }

            System.out.println("vocabMap生成！"); // <位置，vocab>

            List<String> propVocab = Files.readAllLines(Paths.get(propVocabPath));
            propVocabMap = new HashMap<>();
            for (int i = 0; i < propVocab.size(); i += 1) {
                propVocabMap.put(propVocab.get(i), i);
            }

            System.out.println("propVocabMap生成！"); // <位置，propVocab>

            byte[] graphDef = Files.readAllBytes(Paths.get(modelPath));
            Graph graph = new Graph();
            graph.importGraphDef(graphDef);
            session = new Session(graph); //

            System.out.println("开始warm up！");
            warmUp();

        } catch (Exception e) {
            //logger.error("Model init fail.", e);
            //throw e;
            System.out.println("Model init fail " + e);
        }

    }

    //@PostConstruct
    public Classification(String modelPath, String vocabPath, String propVocabPath) throws Exception {
        System.out.println("开始init！");

        final String MODEL_DATA_DIR = "/Users/jing/Downloads/wm/ds/qc_fix_bug_2/";

        final String MODEL_PATH = MODEL_DATA_DIR + "frozen_graph.pb";
        final String VOCAB_PATH = MODEL_DATA_DIR + "in_vocab_10000000.txt";
        final String PROP_VOCAB_PATH = MODEL_DATA_DIR + "property_vocab_10000000.txt";
        final String TASK_LABEL_PATH = MODEL_DATA_DIR + "joint_pattern.txt";
        final String UM_LABEL_PATH = MODEL_DATA_DIR + "manual_pattern.txt";
        final String QA_LABEL_PATH = MODEL_DATA_DIR + "qa_pattern.txt";
        final String CID_LABEL_PATH = MODEL_DATA_DIR + "cid2_pattern.txt";
        final String OTHER_LABEL_PATH = MODEL_DATA_DIR + "other_pattern.txt";
        final String KG_LABEL_PATH = MODEL_DATA_DIR + "kg_pattern.txt";

        String[] paths = {TASK_LABEL_PATH, UM_LABEL_PATH, QA_LABEL_PATH, KG_LABEL_PATH, OTHER_LABEL_PATH};


        for (String path : paths) {
            labelsList.add(Files.readAllLines(Paths.get(path)));
        }

        System.out.println("labelList生成完毕！");
        cid2Labels = Files.readAllLines(Paths.get(CID_LABEL_PATH));

        System.out.println("开始superInit！");
        superInit(MODEL_PATH, VOCAB_PATH, PROP_VOCAB_PATH);

    }

    protected List<String> getFetchNames() {
        List<String> ret = new ArrayList<>();
        // Order matters.
        ret.add(JOINT_PROBABILITIES_NODE); //任务分类
        ret.add(MANUAL_PATTERN_PROBABILITIES_NODE); //用户手册分类
        ret.add(QA_PATTERN_PROBABILITIES_NODE); //闲聊分类
        ret.add(KG_PATTERN_PROBABILITIES_NODE); //百科问答分类
        ret.add(OTHER_PATTERN_PROBABILITIES_NODE); //其他分类
        ret.add(CID2_PATTERN_PROBABILITIES_NODE);
        return ret;
    }


}

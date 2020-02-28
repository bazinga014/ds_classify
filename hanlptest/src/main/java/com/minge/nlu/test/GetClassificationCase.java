package com.minge.nlu.test;

import com.minge.nlu.entity.ClassificationCase;
import com.minge.nlu.module.Classification;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

public class GetClassificationCase {

    private String classficationTestFile = "/Users/jing/Downloads/mg_3/call_data_queryclassify/test/test.seq.in";
    private String propertyMapFile = "/Users/jing/Downloads/mg_3/call_data_queryclassify/prop_info.txt";
    private Map<String, List<String>> propertyMap;
    private List<ClassificationCase> cases = new ArrayList<>();

    public List<ClassificationCase> getCases() {
        return cases;
    }

    public GetClassificationCase() throws IOException {
        readPropertyMap();
        readClassificationTestCases();
    }

    private void readPropertyMap() throws IOException {
        List<String> propertyInfoList = Files.readAllLines(Paths.get(propertyMapFile));
        propertyMap = new HashMap<>();
        for (int i = 1; i < propertyInfoList.size(); i ++) {
            List<String> propertyList = Arrays.asList(propertyInfoList.get(i).split("\t"));
            String word = propertyList.get(1);
            String property = "<" + propertyList.get(2) + ">";
            if (propertyMap.get(word) == null) {
                List<String> value = new ArrayList<>();
                value.add(property);
                propertyMap.put(word, value);
            } else {
                propertyMap.get(word).add(property);
            }
        }
    }

    private void readClassificationTestCases() throws IOException {
        List<String> querys = Files.readAllLines(Paths.get(classficationTestFile));
        for (String query : querys) {
            List<String> words = Arrays.asList(query.split(" "));
            List<String> properties = new ArrayList<>();
            for (String word : words) {
                if (propertyMap.get(word) == null) {
                    properties.add("<blank>");
                } else {
                    List<String> propertyList = propertyMap.get(word);
                    properties.add(String.join(":", propertyList));
                }
            }

            ClassificationCase cCase = new ClassificationCase(words, properties);
            cases.add(cCase);
        }
    }
}

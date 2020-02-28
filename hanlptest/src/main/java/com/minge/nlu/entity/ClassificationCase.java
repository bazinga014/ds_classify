package com.minge.nlu.entity;

import com.minge.nlu.module.Classification;

import java.io.ObjectStreamConstants;
import java.util.List;

public class ClassificationCase {

    private List<String> words;
    private List<String> properties;

    public ClassificationCase(List<String> words, List<String> properties) {
        this.words = words;
        this.properties = properties;
    }

    public void setWords(List<String> words) {
        this.words = words;
    }

    public List<String> getWords() {
        return words;
    }

    public List<String> getProperties() {
        return properties;
    }

    public void setProperties(List<String> properties) {
        this.properties = properties;
    }

}

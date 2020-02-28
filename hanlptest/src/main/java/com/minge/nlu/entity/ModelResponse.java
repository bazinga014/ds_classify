package com.minge.nlu.entity;

import java.util.List;

public class ModelResponse {

    public Integer multiRoundLabel;//1代表是多轮

    public Integer domain;

    public List<String> probabilities;

    public List<String> MRProbabilities;

    public Integer quesType;

    public List<String> result;

    public List<String> slotIntentTagging;

    public List<String> contextSlots;

    public List<Integer> intents;

    public List<String> intentProbabilities;

    public List<String> intentNames;

    public Float mnlp;

    public List<Integer> classifications;

    public List<String> classificationProbabilities;

    public String cid2;

}

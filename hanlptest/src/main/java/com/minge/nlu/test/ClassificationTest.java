package com.minge.nlu.test;

import com.minge.nlu.entity.ClassificationCase;
import com.minge.nlu.entity.ModelResponse;
import com.minge.nlu.module.Classification;

import java.util.List;

public class ClassificationTest {

    private Classification classification;
    private List<ClassificationCase> cCases;

    public ClassificationTest(List<ClassificationCase> cCases) throws Exception {
        classification = new Classification("", "", "");
        this.cCases = cCases;
    }

    public void run() {

        for (ClassificationCase cCase : cCases) {
            ModelResponse ret = classification.ner(cCase.getWords(), cCase.getProperties());
            System.out.println(ret.classifications);
            System.exit(-1);
        }
    }

}

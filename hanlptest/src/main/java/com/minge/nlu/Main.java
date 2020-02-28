package com.minge.nlu;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.seg.common.Term;
import com.hankcs.hanlp.tokenizer.IndexTokenizer;
import com.hankcs.hanlp.tokenizer.NLPTokenizer;
import com.hankcs.hanlp.tokenizer.StandardTokenizer;
import com.minge.nlu.entity.ClassificationCase;
import com.minge.nlu.module.Classification;
import com.minge.nlu.module.*;
import com.minge.nlu.test.ClassificationTest;
import com.minge.nlu.test.GetClassificationCase;

import java.util.List;

public class Main {
    public static void main(String[] args) throws Exception {
        //System.out.println(NLPTokenizer.segment("我新造一个词叫幻想乡你能识别并标注正确词性吗？"));
        System.out.println("开始！");
        //Classification classification = new Classification("", "", "");

        //todo: 获取case
        GetClassificationCase getClassificationCase = new GetClassificationCase();
        List<ClassificationCase> cCases = getClassificationCase.getCases(); //cCases.words = words;cCases.properites = properites

        //todo: 测试和统计
        ClassificationTest classificationTest = new ClassificationTest(cCases);

        long startTime = System.currentTimeMillis();
        classificationTest.run();
        long endTime = System.currentTimeMillis();
        System.out.println("程序运行时间：" + (endTime - startTime) + "ms");
    }
}

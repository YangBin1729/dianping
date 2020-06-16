# 文本细粒度分类
用户评论的细粒度分类，共6大类，20小类。语料来源 https://github.com/AIChallenger/AI_Challenger_2018/tree/master/Baselines/sentiment_analysis2018_baseline。利用 d3.js 可视化预测结果。







![](./clf.gif)





## 模型

模型详情常见：https://github.com/wxue004cs/GCAE   

训练过程常见 :[notebooks](./notebooks)   

将模型文件保存在项目的 `./saved` 文件夹中

## 运行
```shell
> cd dianping
> python app.py
```

访问链接：127.0.0.1/classify
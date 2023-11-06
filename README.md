This repository is related to the paper [Fine-Tuning is not (Always) Overfitting Artifacts](https://perso.uclouvain.be/fstandae/PUBLIS/303.pdf)

## Code ##
To run the code, launch the notebook "Code esann.ipynb". The structure of the code is the same as the one of the paper.  
To launch it on my workstation, i usually run: 
```CUBLAS_WORKSPACE_CONFIG=:16:8 PYTHONHASHSEED=42 nice jupyter notebook```

## Datasets ##
The available datasets are:
  1) Allocine - Movie review classification (https://huggingface.co/datasets/allocine)
  2) InfOpinion - Opinion classification (InfOpinion_dataset.csv)

The InfOpinion dataset was built based on the RTBF dataset presented in [The RTBF Corpus: a dataset of 750,000 Belgian French news articles published between 2008 and 2021](https://dial.uclouvain.be/pr/boreal/object/boreal:276580) and available [here](https://dataverse.uclouvain.be/dataset.xhtml?persistentId=doi:10.14428/DVN/PEVSSI). It was preprocessed using the method presented in [TIPECS : A corpus cleaning method using machine learning and qualitative analysis](https://dial.uclouvain.be/pr/boreal/object/boreal:276581). If you're interested in studying french speaking press, don't hesitate to take a look at it ! :) 

To change from dataset to another, go to cell number 4.

## Explanation and visualization ##
The explanations are computed using the method presented in [Transformer interpretability beyond attention visualization](https://openaccess.thecvf.com/content/CVPR2021/papers/Chefer_Transformer_Interpretability_Beyond_Attention_Visualization_CVPR_2021_paper.pdf) and available [here](https://github.com/hila-chefer/Transformer-Explainability). We adapted it to our model in the "BERT_explainability" folder. 

The visualizations are provided by the method presented in [Fast Multiscale Neighbor Embedding](https://ieeexplore.ieee.org/document/9308987) and available [here](https://github.com/cdebodt/Fast_Multi-scale_NE). 

## Cite our paper ##
If you make use of our work, please cite our paper:

```
@InProceedings{Bogaert_2023_ESANN,
    author    = {Bogaert, Jeremie and Jean, Emmanuel and De Bodt, Cyril and Standaert, François-Xavier},
    title     = {Fine-tuning is not (always) overfitting},
    booktitle = {Proceedings of the 31th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning},
    month     = {October},
    year      = {2023},
    pages     = {}
}
```


# A graph transformer defence against graph perturbation by a flexible-pass filter
Introducing the Flexible-pass Filter-based Graph Transformer (FFGT), a robust defense mechanism against adversarial attacks on graph data. Leveraging the inherent capability of self-attention to adopt diverse graph filters, we've constructed a self-attention layer with three heads, each targeting different frequency ranges: low, hybrid, and high frequencies. To enhance the efficacy of self-attention, we've incorporated graph learning and learning-based fusion modules, yielding a versatile frequency representation. As a result, FFGT showcases consistent performance in thwarting adversarial graph perturbations across different datasets and attack scenarios.

he paper can be viewed via file: https://www.sciencedirect.com/science/article/pii/S1566253524000745. 
![image](utils/data/github.png)
and the framework of our FFGT is listed in the following.

We prodive three demos of main function, which is with respect to three different attacks. Parameters of optimal performance are different across datasets and attacks methods.

# Other:
If you are interested in our work, please also cite our paper: 
@article{zhu2024graph,
  title={Graph transformer against graph perturbation by flexible-pass filter},
  author={Zhu, Yonghua and Huang, Jincheng and Chen, Yang and Amor, Robert and Witbrock, Michael},
  journal={Information Fusion},
  pages={102296},
  year={2024},
  publisher={Elsevier}
}
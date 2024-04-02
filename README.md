# A graph transformer defence against graph perturbation by a flexible-pass filter
Introducing the Flexible-pass Filter-based Graph Transformer (FFGT), a robust defense mechanism against adversarial attacks on graph data. Leveraging the inherent capability of self-attention to adopt diverse graph filters, we've constructed a self-attention layer with three heads, each targeting different frequency ranges: low, hybrid, and high frequencies. To enhance the efficacy of self-attention, we've incorporated graph learning and learning-based fusion modules, yielding a versatile frequency representation. As a result, FFGT showcases consistent performance in thwarting adversarial graph perturbations across different datasets and attack scenarios.

The paper can be viewed via the following file: https://www.sciencedirect.com/science/article/pii/S1566253524000745. 
The framework of our FFGT is listed below.
![image](https://github.com/yhzhu66/FFGT/assets/52006047/f1526322-5fd3-4e46-8b97-db18d27ffd50)


We produce three demos of the main function to run our code, which is with respect to three different attacks. Parameters of optimal performance are different across datasets and attack methods.

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

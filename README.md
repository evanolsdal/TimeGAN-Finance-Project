# TimeGAN-Finance-Project

This repository serves as a platform to demonstrate the practical implementation of the Time Series Generative Adversarial Network (TimeGAN), a cutting-edge architecture derived from the paper "Time-series Generative Adversarial Networks" by Yoon et al. [Read Paper Here](https://proceedings.neurips.cc/paper_files/paper/2019/file/c9efe5f26cd17ba6216bbe2a7d26d490-Paper.pdf)

TimeGAN represents a sophisticated synergy between the traditional generative neural network structure of a GAN and an unique supervised learning framework. This integration facilitates a more refined and comprehensive approach to the generation of time series data. The GAN's unsupervised component contributes realism and novel variability to the generated data. Meanwhile, the supervised element ensures that the model preserves the temporal dynamics of the real data. This is particularly critical as the Discriminator's loss function in conventional architectures can sometimes obscure these essential temporal dynamics.

Although the TimeGAN developed here is versatile enough to be applied to the generation of diverse time series data, this repository primarily focuses on its application within the realm of financial time series. Because of this there are several tailor made functions and network architecture options to optimize the modeling of financial data. Nevertheless, these components can be easily adapted and fine-tuned to suit the broader spectrum of time series applications.

# Digital Twin Aided Compressive Sensing: Enabling Site-Specific MIMO Hybrid Precoding
  This is a python code package related to the following article: H.Luo and A. Alkhateeb, "[Digital Twin Aided Compressive Sensing: Enabling Site-Specific MIMO Hybrid Precoding](https://www.wi-lab.net/research/digital-twin-aided-compressive-sensing/)", accepted to 58th Asilomar Conference on Signals, Systems, and Computers, 2024.

# Abstract of the Article
<div align="justify">Compressive sensing is a promising solution for the channel estimation in multiple-input multiple-output (MIMO) systems with large antenna arrays and constrained hardware. Utilizing site-specific channel data from real-world systems, deep learning can be employed to learn the compressive sensing measurement vectors with minimum redundancy, thereby focusing sensing power on promising spatial directions of the channel. Collecting real-world channel data, however, is challenging due to the high overhead resulting from the large number of antennas and hardware constraints. In this paper, we propose leveraging a site-specific digital twin to generate synthetic channel data, which shares a similar distribution with real-world data. The synthetic data is then used to train the deep learning models for learning measurement vectors and hybrid precoder/combiner design in an end-to-end manner. We further propose a model refinement approach to fine-tune the model pre-trained on the digital twin data with a small amount of real-world data. The evaluation results show that, by training the model on the digital twin data, the learned measurement vectors can be efficiently adapted to the environment geometry, leading to high performance of hybrid precoding for real-world deployments. Moreover, the model refinement approach can enable the digital twin aided model to achieve comparable performance to the model trained on the real-world dataset with a significantly reduced amount of real-world data.</div>

# Code Package Content

**Prepare the dataset**
1. Download DeepVerse scenario
```
python DeepVerse_downloader.py
```
2. Generate dataset
```
python DeepVerse_generator.py
```

**ML Model Training**
1. Generate training and testing datasets
```
python gen_csv.py
```
2. Run the training session
```
python train.py
```

**Plot the results**
1. Obtain the measurement vectors from the model weights.
```
python inference.py --load_model_path ckpt/ckpt_name
```
2. Plot the beam patterns.
```
python plot_meas_vecs.py
```

If you have any questions regarding the code, please contact [Hao Luo](mailto:h.luo@asu.edu)

# License and Referencing
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This code package is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/).

If you in any way use this code for research that results in publications, please cite our original article:
> H. Luo and A. Alkhateeb, “Digital Twin Aided Compressive Sensing: Enabling Site-Specific MIMO Hybrid Precoding,” arXiv preprint arXiv:2405.07115, 2024.

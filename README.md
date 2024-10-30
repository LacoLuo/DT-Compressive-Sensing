# Digital Twin Aided Compressive Sensing: Enabling Site-Specific MIMO Hybrid Precoding
  This is a python code package related to the following article: H.Luo and A. Alkhateeb, "[Digital Twin Aided Compressive Sensing: Enabling Site-Specific MIMO Hybrid Precoding](https://www.wi-lab.net/research/digital-twin-aided-compressive-sensing/)", accepted to 58th Asilomar Conference on Signals, Systems, and Computers, 2024.

# Abstract of the Article
<div align="justify">Compressive sensing is a promising solution for the channel estimation in multiple-input multiple-output (MIMO) systems with large antenna arrays and constrained hardware. Utilizing site-specific channel data from real-world systems, deep learning can be employed to learn the compressive sensing measurement vectors with minimum redundancy, thereby focusing sensing power on promising spatial directions of the channel. Collecting real-world channel data, however, is challenging due to the high overhead resulting from the large number of antennas and hardware constraints. In this paper, we propose leveraging a site-specific digital twin to generate synthetic channel data, which shares a similar distribution with real-world data. The synthetic data is then used to train the deep learning models for learning measurement vectors and hybrid precoder/combiner design in an end-to-end manner. We further propose a model refinement approach to fine-tune the model pre-trained on the digital twin data with a small amount of real-world data. The evaluation results show that, by training the model on the digital twin data, the learned measurement vectors can be efficiently adapted to the environment geometry, leading to high performance of hybrid precoding for real-world deployments. Moreover, the model refinement approach can enable the digital twin aided model to achieve comparable performance to the model trained on the real-world dataset with a significantly reduced amount of real-world data.</div>

# Code Package Content

**Prepare the dataset**
1. The data used in this package can be found in this [Dropbox folder](https://www.dropbox.com/scl/fo/5u29i71qptn23wvykb88d/AA-9db1geL73lkdlH6gqx3o?rlkey=v8bvb2kdx5nayc12yt2d4ahpt&st=b03julkg&dl=0). Please download these files to the `DeepMIMO` repository.
2. Set the scenario and other system parameters in `parameters.m`.
3. Run `DeepMIMO_Dataset_Generator.m` to generate channel data of the scenario.
4. Run `process_raw_data.m` to construct the dataset.

**ML Model Training**
1. Generate training and testing datasets
```
python gen_csv.py
```
2. Run the training sessions with varying numbers of measurement vectors
```
python train_loop_dict_size.py
```
3. Refine the models pretrained on DT data
```
python train_loop_dict_size_finetune.py
```

**Plot the results**
1. Plot the RF beam prediction accuracy vs. numbers of measurement vectors
```
python plot_performance_dict_size.m
```
2. Plot the RF beam prediction accuracy vs. numbers of refining data points
```
python plot_performance_num_data.m
```
3. Plot the beam patterns of the learned measurement vectors
   - Obtain the measurement vectors from the model weights.
   ```
   python inference.py --load_model_path ckpt/ckpt_name
   ```
   - Run `plot_meas_vecs.m` to plot the beam patterns.

If you have any questions regarding the code, please contact [Hao Luo](mailto:h.luo@asu.edu)

# License and Referencing
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This code package is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/).

If you in any way use this code for research that results in publications, please cite our original article:
> H. Luo and A. Alkhateeb, “Digital Twin Aided Compressive Sensing: Enabling Site-Specific MIMO Hybrid Precoding,” arXiv preprint arXiv:2405.07115, 2024.

# Triple-Space Representation Learning for MSA
Reference :Code for the [ACM MM 2020](https://2020.acmmm.org) paper [MISA: Modality-Invariant and -Specific Representations for Multimodal Sentiment Analysis](https://arxiv.org/pdf/2005.03545.pdf)


<p align="center">
  <img width="600" src="misa-pic.png">
</p>



### Setup the environment

We work with a conda environment.

```
conda env create -f environment.yml
conda activate misa-code
```

### Data Download

- Install [CMU Multimodal SDK](https://github.com/A2Zadeh/CMU-MultimodalSDK). Ensure, you can perform ```from mmsdk import mmdatasdk```.    
- Option 1: Download [pre-computed splits](https://drive.google.com/drive/folders/1IBwWNH0XjPnZWaAlP1U2tIJH6Rb3noMI?usp=sharing) and place the contents inside ```datasets``` folder.     
- Option 2: Re-create splits by downloading data from MMSDK. For this, simply run the code as detailed next.

### Running the code

1. ```cd src```
2. Set ```word_emb_path``` in ```config.py``` to [glove file](http://nlp.stanford.edu/data/glove.840B.300d.zip).
3. Set ```sdk_dir``` to the path of CMU-MultimodalSDK.
2. ```python train.py --use_facet_visual True --data mosi --name "test_facet" --patience 20```. Replace ```mosi``` with ```mosei``` or ```ur_funny``` for other datasets.
3. ```python train.py --use_facet_visual True --data mosei --name "test_facet" --learning_rate 1e-5 --facet_visual_size 35```

### Citation

If this paper is useful for your research, please cite us at:

```
@article{,
  title={Triple-Space Representation Learning for MSA},
  author={},
  journal={},
  year={2025}
}
```

### Contact

For any questions, please email at 

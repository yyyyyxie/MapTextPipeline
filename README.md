# Introduction

ICDAR https://rrc.cvc.uab.es/
This competition is internationally recognized as an authoritative event in the field of text recognition. The data evaluation and metrics in top conference papers in the text recognition field often come from ICDAR competition data and metrics. Generally, there are several major events each year, and each event is further divided into 3-4 competitions.

## Introduction to the ICDAR24 Competition on Historical Map Text Detection, Recognition, and Linking

https://rrc.cvc.uab.es/?ch=28&com=introduction

Text on digitized historical maps contains valuable information providing georeferenced political and cultural context, yet the wealth of information in digitized historical maps remains largely inaccessible due to their unsearchable raster format. This competition aims to address the unique challenges of **detecting and recognizing** textual information (e.g., place names) and **linking** words to form location phrases.

While the *detection* and *recognition* tasks share similarities with the long line of prior robust reading competitions [1,2], historical map text extraction presents challenges such as dense text regions, rotated and curved text and widely spaced characters which are not very common in scene text extraction problems. The word *linking* task, in particular, is quite challenging as words can be highly spaced with complicated text-like distractors, even other words appearing between the characters. Furthermore, words within a single location phrase may be divided across multiple lines to optimize label placement. The figure below illustrates primary challenges.

![map_picture_2.jpg](https://rrc.cvc.uab.es/files/map_picture_2.jpg)



# Usage


## Usage

- ### Installation

Python 3.8 + PyTorch 2.0.1 + CUDA 11.7 + Detectron2

```
conda create -n dnts python=3.8 -y
conda activate dnts
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
cd detectron2
pip install -e .
pip install -r requirements.txt
cd ..
python setup.py build develop
```

- ## Convert to COCO formats:

```
python tools/convert.py --input-json your_gt_path --output-json your_output_path --output_image_id_json your_output_image_id_path
```



- ## Convert results to submition formats:

```
python tools/convert_to_original.py --input-json your_pred_path  --input_image_id_json your_input_image_id_path --output-json your_output_submition_format_path 
```



### Fine-tune

You can download our pre-trained model in [OneDrive](https://drive.google.com/file/d/13rPnEcWu2FGwGw1BgH0UAw8LS2gu2RYK/view?usp=drive_link) and fine-tune it on the Rumsey dataset. The fine-tuning command is as follows:

```
python tools/train.py --config-file configs/ViTAEv2_S/rumsey/final_rumsey.yaml --num-gpus 2
```

You can also directly use our fine-tuned [weights](https://drive.google.com/file/d/1Okvl5tlWusJxDCdDv_CLsGKQ5elImfx4/view?usp=drive_link)  for inference:

```
python tools/train.py --config-file configs/ViTAEv2_S/rumsey/test.yaml --num-gpus 2 --eval-only
```

JSON results will be saved in output/vitaev2/test/rumsey_bs2_test_final/inference/text_results.json, and you can use tools/convert_to_original.py to convert the JSON file to submission results.



## Citation

This project utilizes methods related to [DNTextSpotter](https://github.com/yyyyyxie/DNTextSpotter). If you find MapTextPipeline helpful, please consider giving this repo a star ⭐ and citing:



## Acknowledgement

This project is based on [Adelaidet](https://github.com/aim-uofa/AdelaiDet) and [DeepSolo](https://github.com/ViTAE-Transformer/DeepSolo). For academic use, this project is licensed under the 2-clause BSD License.


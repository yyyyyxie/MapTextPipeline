Convert to COCO formats:

python tools/convert.py --input-json your_gt_path --output-json your_output_path --output_image_id_json your_output_image_id_path



Convert results to submition formats:

python tools/convert_to_original.py --input-json your_pred_path  --input_image_id_json your_input_image_id_path --output-json your_output_submition_format_path 



## pretrain

python tools/train.py --config-file configs/ViTAEv2_S/pretrain/rumsey_pretrain.yaml --num-gpus 4



### finetune

python tools/train.py --config-file configs/ViTAEv2_S/rumsey/final_rumsey.yaml --num-gpus 2

Json results will be saved in output/vitaev2/finetune/a40_rumsey_bs2_final_add_val_final/inference/text_results.json

time python3 main.py --mode 'train' --blobs_folder_path '../JIGSAWS/Suturing/blobs' --weights_save_path models --model_dim 2048

time python main.py --mode eval --blobs_path '../JIGSAWS/Suturing/blobs/' --weights_save_path 'models/multimodal_Suturing_2022-11-02_05:08:13.pth' --transcriptions_path '../JIGSAWS/Suturing/transcriptions' --experimental_setup_path '../JIGSAWS/Experimental_setup/Suturing/Balanced/GestureClassification/UserOut/1_Out' --model_dim 2048

rm imgs/* && time python main.py --mode umap --blobs_path '../JIGSAWS/Suturing/blobs/'  --weights_save_path 'models/multimodal_Suturing_2022-11-02_05:08:13.pth' --model_dim 2048 && python finish-plots.py


# moco (replace pretrained_encoder_weights with the path to your encoder weights from hw)
time python3 main.py --mode 'train-moco' --blobs_folder_path '../JIGSAWS/Suturing/blobs' --weights_save_path moco-models --model_dim 2048 --pretrained_encoder_weights 'models/multimodal_Suturing_2022-11-02_05:08:13.pth'


#contrastive
time python3 main.py --mode 'train-contrastive' --blobs_folder_path '../JIGSAWS/Suturing/blobs' --weights_save_path models-contrastive --model_dim 2048 --noise_variance 1e-8


#kinematics contrastive
time python3 main.py --mode 'train-contrastive-kin' --blobs_folder_path '../JIGSAWS/Suturing/blobs' --weights_save_path models-contrastive-kin --model_dim 2048 --noise_variance 1e-8

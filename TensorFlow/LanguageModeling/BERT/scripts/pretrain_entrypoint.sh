git clone --branch feature/bert_numeracy https://github.com/kenhktsui/DeepLearningExamples.git
cd DeepLearningExamples/TensorFlow/LanguageModeling/BERT
# Establish container
bash scripts/docker/build.sh
# Download, preprocess, tokenize data; download model pretrained weight
bash scripts/data_download.sh wiki_only
# Enter container
bash scripts/docker/launch.sh
# Training
bash scripts/run_pretraining_adam.sh 32 8 1e-4 fp32 false 1 10000 1144000 5000 base 1 128 20 wikicorpus_en

#Extract ApollMoE Data to Major or Minor part
python metadata/extract_dataset.py \
--dataset_path /path/to/ApolloMoEDataset.json \
--major_dataset_path /path/to/MajorDataset.json \
--minor_dataset_path /path/to/MinorDataset.json

python metadata/extract_bench.py \
--bench_path /path/to/ApolloMoEBench.json \
--major_bench_path /path/to/MajorBench.json \
--minor_bench_path /path/to/MinorBench.json

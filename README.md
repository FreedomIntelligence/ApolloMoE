# Democratizing Medical LLMs For Much More Languages

Covering 12 Major Languages including English, Chinese, French, Hindi, Spanish, Arabic, Russian, Japanese, Korean, German, Italian, Portuguese and 38 Minor Languages So far.
<center>



<p align="center">
   📃 <a href="https://arxiv.org/abs/2410.10626" target="_blank">Paper</a> • 🌐 <a href="" target="_blank">Demo</a> • 🤗 <a href="https://huggingface.co/datasets/FreedomIntelligence/ApolloMoEDataset" target="_blank">ApolloMoEDataset</a> • 🤗 <a href="https://huggingface.co/datasets/FreedomIntelligence/ApolloMoEBench" target="_blank">ApolloMoEBench</a>  • 🤗 <a href="https://huggingface.co/collections/FreedomIntelligence/apollomoe-and-apollo2-670ddebe3bb1ba1aebabbf2c" target="_blank">Models</a> • 🌐 <a href="https://github.com/FreedomIntelligence/Apollo" target="_blank">Apollo</a>
</p>

![Apollo](assets/apollo_medium_final.png)

## 🌈 Update

* **[2024.10.15]** ApolloMoE repo is published！🎉


## Languages Coverage
12 Major Languages and 38 Minor Languages

<details>
  <summary>Click to view the Languages Coverage</summary>
   
   ![ApolloMoE](assets/languages.png)

</details>


## Architecture

<details>
  <summary>Click to view the MoE routing image</summary>

  ![ApolloMoE](/assets/hybrid_routing.png)

</details>

## Results

### Dense
   🤗 <a href="https://huggingface.co/FreedomIntelligence/Apollo2-0.5B" target="_blank">Apollo2-0.5B</a> • 🤗 <a href="https://huggingface.co/FreedomIntelligence/Apollo2-1.5B" target="_blank">Apollo2-1.5B</a> • 🤗 <a href="https://huggingface.co/FreedomIntelligence/Apollo2-2B" target="_blank">Apollo2-2B</a>  • 🤗 <a href="https://huggingface.co/FreedomIntelligence/Apollo2-3.8B" target="_blank">Apollo2-3.8B</a> • 🤗 <a href="https://huggingface.co/FreedomIntelligence/Apollo2-7B" target="_blank">Apollo2-7B</a>  • 🤗 <a href="https://huggingface.co/FreedomIntelligence/Apollo2-9B" target="_blank">Apollo2-9B</a>  
   
<details>
  <summary>Click to view the Dense Models Results</summary>
   
   ![ApolloMoE](assets/dense_results.png)

</details>

### Post-MoE
   🤗 <a href="https://huggingface.co/FreedomIntelligence/Apollo-MoE-0.5B" target="_blank">Apollo-MoE-0.5B</a>  • 🤗 <a href="https://huggingface.co/FreedomIntelligence/Apollo-MoE-1.5B" target="_blank">Apollo-MoE-1.5B</a>  • 🤗 <a href="https://huggingface.co/FreedomIntelligence/Apollo-MoE-7B" target="_blank">Apollo-MoE-7B</a>  
   
<details>
  <summary>Click to view the Post-MoE Models Results</summary>
   
   ![ApolloMoE](assets/post_moe_results.png)

</details>


## Usage Format
#### Apollo2
- 0.5B, 1.5B, 7B: User:{query}\nAssistant:{response}<|endoftext|>
- 2B, 9B: User:{query}\nAssistant:{response}\<eos\>
- 3.8B: <|user|>\n{query}<|end|><|assisitant|>\n{response}<|end|>

#### Apollo-MoE
- 0.5B, 1.5B, 7B: User:{query}\nAssistant:{response}<|endoftext|>

## Dataset & Evaluation

- Dataset
  🤗 <a href="https://huggingface.co/datasets/FreedomIntelligence/ApolloMoEDataset" target="_blank">ApolloMoEDataset</a>

   <details><summary>Click to expand</summary>

    ![ApolloMoE](assets/Dataset.png)

    - [Data category](https://huggingface.co/datasets/FreedomIntelligence/ApolloCorpus)


   </details>


   The complete data is stored in `ApolloMoEDataset.json`, while a sample shown in `ApolloMoEDataset_sample.json`
- Evaluation
  🤗 <a href="https://huggingface.co/datasets/FreedomIntelligence/ApolloMoEBench" target="_blank">ApolloMoEBench</a> 

   <details><summary>Click to expand</summary>
      
     - EN:
       - [MedQA-USMLE](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options) 
       - [MedMCQA](https://huggingface.co/datasets/medmcqa/viewer/default/test)
       - [PubMedQA](https://huggingface.co/datasets/pubmed_qa): Because the results fluctuated too much, they were not used in the paper.
       - [MMLU-Medical](https://huggingface.co/datasets/cais/mmlu)
         - Clinical knowledge, Medical genetics, Anatomy, Professional medicine, College biology, College medicine
     - ZH:
       - [MedQA-MCMLE](https://huggingface.co/datasets/bigbio/med_qa/viewer/med_qa_zh_4options_bigbio_qa/test)
       - [CMB-single](https://huggingface.co/datasets/FreedomIntelligence/CMB): Not used in the paper
         - Randomly sample 2,000 multiple-choice questions with single answer.
       - [CMMLU-Medical](https://huggingface.co/datasets/haonan-li/cmmlu)
         - Anatomy, Clinical_knowledge, College_medicine, Genetics, Nutrition, Traditional_chinese_medicine, Virology
       - [CExam](https://github.com/williamliujl/CMExam): Not used in the paper
         - Randomly sample 2,000 multiple-choice questions


     - ES: [Head_qa](https://huggingface.co/datasets/head_qa)
     - FR:
       - [Frenchmedmcqa](https://github.com/qanastek/FrenchMedMCQA)
       - [MMLU_FR]
         - Clinical knowledge, Medical genetics, Anatomy, Professional medicine, College biology, College medicine
     - HI: [MMLU_HI](https://huggingface.co/datasets/FreedomIntelligence/MMLU_Hindi)
        - Clinical knowledge, Medical genetics, Anatomy, Professional medicine, College biology, College medicine
     - AR: [MMLU_AR](https://huggingface.co/datasets/FreedomIntelligence/MMLU_Arabic)
        - Clinical knowledge, Medical genetics, Anatomy, Professional medicine, College biology, College medicine
     - JA: [IgakuQA](https://github.com/jungokasai/IgakuQA)
     - KO: [KorMedMCQA](https://huggingface.co/datasets/sean0042/KorMedMCQA)
     - IT:
       - [MedExpQA](https://huggingface.co/datasets/HiTZ/MedExpQA)
       - [MMLU_IT]
         - Clinical knowledge, Medical genetics, Anatomy, Professional medicine, College biology, College medicine
     - DE: [BioInstructQA](https://huggingface.co/datasets/BioMistral/BioInstructQA): German part
     - PT: [BioInstructQA](https://huggingface.co/datasets/BioMistral/BioInstructQA): Portuguese part
     - RU: [RuMedBench](https://github.com/sb-ai-lab/MedBench)

      
      


   </details>

   
## Results reproduction
   
   (Optional) Custom Model as Base
   
   <details><summary>Click to expand</summary>
      
   ```
      copy /path/to/your/configuration_upcycling_qwen2_moe.py /path/to/src/variants/moe_initilization/configuration_upcycling_qwen2_moe.py
      copy /path/to/your/modeling_upcycling_qwen2_moe.py /path/to/src/variants/moe_initilization/modeling_upcycling_qwen2_moe.py
      cd /path/to/src/variants/moe_initilization
      bash convert.sh
   ```

   </details>

   Full-finetune on Base Model
   
   <details><summary>Click to expand</summary>

   
   
   We take Apollo2-7B or Apollo-MoE-0.5B as example

   
   1. Download and extract data:
      
      - Dowload Dataset and Benchmark firstly
      - Extract major or minor data part according to your needs:


      ```
      bash 0.extract_data.sh
      ```   
    
   2. Prepare test and dev data for specific model:
      - Create test data for with special token
        
       ```
       bash 1.data_process_test&dev.sh
       ```
    
   3. Prepare train data for specific model (Create tokenized data in advance):

    
      - You can adjust data Training order and Training Epoch in this step

       ```
       bash 2.data_process_train.sh
       ```
    
   4. Train the model

    
      - If you want to train in Multi Nodes please refer to ./src/sft/training_config/zero_multi.yaml


       ```
       bash 3.single_node_train.sh
       ```


   5. Evaluate your model: Generate score for benchmark
      
         ```
         bash 4.eval.sh
         ```

   </details>



##  Citation
Please use the following citation if you intend to use our dataset for training or evaluation:

```
@misc{zheng2024efficientlydemocratizingmedicalllms,
      title={Efficiently Democratizing Medical LLMs for 50 Languages via a Mixture of Language Family Experts}, 
      author={Guorui Zheng and Xidong Wang and Juhao Liang and Nuo Chen and Yuping Zheng and Benyou Wang},
      year={2024},
      eprint={2410.10626},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.10626}, 
}
```


This repository contains source code and data for the paper " On the Application of Natural Architecture Search for Parameter-Efficient Fine-Tuning of Pre-trained Code Models " 

# Abstract : 

Large Language Models (LLMs) have demonstrated their ability to solve tasks across various domains, including software engineer- ing. However, their extensive number of parameters makes full fine-tuning computationally prohibitive. While parameter-efficient fine-tuning methods, e.g. adapter fine-tuning, have been proposed to address these challenges, they often rely on standard design choices and configuration. Meanwhile, Neural Architecture Search (NAS) methods can successfully optimize neural network archi- tectures and have been employed to automate the configuration of Parameter-Efficient Fine-Tuning (PEFT) methods in other do- mains (e.g. Natural Language Processing). This study investigates the potential of NAS to design custom adapter architectures and configurations for software engineering tasks, i.e. vulnerability detection and code clone detection. We explore diverse adapter modules architectures and evaluate the benefits of selective adapter insertion. Our results turn out to be negative, suggesting that the architecture of the PEFT modules has minimal impact on the fine- tuning process’s performance, and the gains from extended search may not justify the computational overhead. Our experiments still reveal that inserting adapters in earlier layers, increasing the size of adapters, and adapting different submodules of the LLM layer, can improve fine-tuning performance effectively, saving both time and computational resources. Although learned adapter parameters are dataset-specific, our findings suggest the possibility of transferring adapter configurations to similar datasets and tasks, thus simpli- fying the search for optimal PEFT settings. 


# Datasets : 

## Defect detection : 
https://drive.google.com/file/d/1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF/view
## clone detection  : 
https://github.com/clonebench/BigCloneBench

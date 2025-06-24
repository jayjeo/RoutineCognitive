# Research Project Structure and File Descriptions

## Project Overview
This research project analyzes task polarization using Large Language Models (LLMs) to classify occupational tasks as routine/non-routine and cognitive/manual. The project integrates multiple AI models, deep learning approaches, and econometric analysis.

## Replication File Download
- [Dropbox Link](https://www.dropbox.com/scl/fo/96ir1z4hf7yg1wa2haa39/AEtm-ZxSd9eNfd9FRnwtAD0?rlkey=dibdjeinp3f3v4d86dhte4q6r&st=49lfjp4v)

## Requirements
- At least 20 CPUs
- A100 80GB GPU
- At least 64GB RAM
- Windows 10 / 11

## Python Environment Installation
- Python version 3.12.10
- [Install CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- Recommend using conda (Miniconda)
  - conda create -n myenv python=3.12.10
  - conda activate myenv
- At your desired location (downloaded location),
  - pip install -r requirements.txt

## 1. Main Python Files Tree Structure

### 1.1 Data Processing and Classification Pipeline

```
clearcache.py
├── Input: None (system utility)
└── Output: Cleaned cache files

config_absolute_path.py
├── Input: None (configuration file)
└── Output: Path configurations for all scripts

OPUSvalue.py
├── Input: Taskvalue.dta
├── Dependencies: config_absolute_path.py
└── Output: updated_file_final_OPUS.dta

GPTvalue.py
├── Input: Taskvalue.dta
├── Dependencies: config_absolute_path.py
└── Output: updated_file_final_GPT.dta

OPUSvalue_cognitive.py
├── Input: Taskvalue.dta
├── Dependencies: config_absolute_path.py
└── Output: updated_file_final_cognitive_OPUS.dta

GPTvalue_cognitive.py
├── Input: Taskvalue.dta
├── Dependencies: config_absolute_path.py
└── Output: updated_file_final_cognitive_GPT.dta
```

### 1.2 Machine Learning Analysis Pipeline

```
MachineAnalysis/run_analysis.py
├── Input: rou.dta, cog.dta
├── Dependencies: config_absolute_path.py
├── Output: embeddings_mxbai_embed_large_*.pkl, analysis results
└── Workflow: Interactive menu → embedding generation → model training

MachineFigure/deeplearning_score_predictor.py
├── Input: rou.dta, embeddings_*.pkl
├── Dependencies: config_absolute_path.py
├── Output: deeplearning_results.pkl, deeplearning_analysis.png
└── Models: SimpleNet, DeepNet, ResNet

MachineFigure/deeplearning_cognitive_score_predictor.py
├── Input: cog.dta, embeddings_*.pkl
├── Dependencies: config_absolute_path.py
├── Output: deeplearning_results_cognitive.pkl, deeplearning_cognitive_analysis.png
└── Models: SimpleNet, DeepNet, ResNet

MachineFigure/deeplearning_network_viz.py
├── Input: rou.dta, embeddings_*.pkl, deeplearning_results.pkl
├── Dependencies: config_absolute_path.py
└── Output: deeplearning_network_visualization.png
```

### 1.3 Advanced Deep Learning Pipeline

```
Deeplearning/Deepscoring_Enhanced.py
├── Input: rou.dta, deeplearning_results.pkl
├── Dependencies: config_absolute_path.py
├── Components: GemmaReliabilityJudge, ReliabilityRefinementModel, FinalScoreModel
└── Output: Deepresult_Enhanced_*.dta (chunked results)

Deeplearning_cognitive/Deepscoring_Enhanced_cognitive.py
├── Input: cog.dta, deeplearning_results_cognitive.pkl
├── Dependencies: config_absolute_path.py
├── Components: GemmaReliabilityJudge, ReliabilityRefinementModel, FinalScoreModel
└── Output: Deepresult_Enhanced_cognitive_*.dta (chunked results)

gemma3_routine/gemma_processor.py
├── Input: Taskvalue.dta
├── Dependencies: config.yaml (separate config)
├── Components: GemmaProcessor with Gemma-3-27b-it model
└── Output: chunk_*.dta, final aggregated results
```

### 1.4 Random Forest Analysis Pipeline

```
rf_threading.py
├── Input: TasksGPT_SOC2010_CPSready_region_temp.dta
├── Dependencies: None
├── Components: CUDA-optimized Random Forest with parallel processing
└── Output: Enhanced temporal analysis with GPU acceleration
```

### 1.5 Stata Main Analysis

```
main.do
├── Input: Multiple data files (CPS, ONET, classification results)
├── Dependencies: All Python classification outputs
├── Components: Data merging, econometric analysis, visualization
└── Output: Multiple graphs (.eps), regression tables (.tex), processed datasets
```

## 2. File Descriptions and Functions

### 2.1 Utility and Configuration Files

#### clearcache.py
**Role**: System maintenance utility for the deep learning project  
**Function**: 
- Safely cleans project-specific cache files that may affect analysis results
- Preserves critical files (model downloads, .pkl results, source code)
- Targets specific cache patterns (__pycache__, .mypy_cache, etc.)
- Interactive safety confirmation before deletion

#### config_absolute_path.py
**Role**: Central path configuration manager  
**Function**: 
- Defines all absolute paths used across the project
- Provides single point of configuration for different computer setups
- Manages paths for data files, embeddings, results, and output directories
- Includes path validation functionality

### 2.2 LLM Classification Files

#### OPUSvalue.py
**Role**: Task classification using Claude OPUS model  
**Function**: 
- Classifies ONET tasks as routine (0) or non-routine (1)
- Uses Anthropic's Claude-3-opus-20240229 model
- Processes tasks in parallel with ThreadPoolExecutor
- Implements chunked processing with resume capability
- Outputs detailed reasoning for each classification

#### GPTvalue.py
**Role**: Task classification using GPT-4 model  
**Function**: 
- Performs same routine/non-routine classification as OPUS
- Uses OpenAI's GPT-4 model for comparison
- Sequential processing with retry mechanisms
- Chunked output for large-scale processing

#### OPUSvalue_cognitive.py
**Role**: Cognitive task classification using Claude OPUS  
**Function**: 
- Classifies tasks as cognitive (1) or manual (0)
- Focuses on mental vs. physical task requirements
- Parallel processing with enhanced threading
- Detailed cognitive reasoning analysis

#### GPTvalue_cognitive.py
**Role**: Cognitive task classification using GPT-4  
**Function**: 
- Parallel cognitive classification to OPUS version
- Enables comparison between GPT-4 and Claude OPUS on cognitive dimensions
- Sequential processing with error handling

### 2.3 Machine Learning Analysis Files

#### MachineAnalysis/run_analysis.py
**Role**: Interactive machine learning pipeline controller  
**Function**: 
- Provides user-friendly menu for analysis options
- Manages embedding generation and model training
- Implements safe Random Forest parameters to prevent long execution times
- Includes process management and optimization features
- Handles existing data recovery and continuation

#### MachineFigure/deeplearning_score_predictor.py
**Role**: Deep learning model for routine score prediction  
**Function**: 
- Trains multiple neural network architectures (SimpleNet, DeepNet, ResNet)
- Uses text embeddings to predict OPUS-GPT score differences
- Implements feature extraction from intermediate layers
- Generates comprehensive performance visualizations
- CUDA-optimized for GPU acceleration

#### MachineFigure/deeplearning_cognitive_score_predictor.py
**Role**: Deep learning model for cognitive score prediction  
**Function**: 
- Parallel implementation to routine score predictor for cognitive tasks
- Trains on cognitive score differences between OPUS and GPT
- Same neural network architectures with cognitive-specific tuning
- Generates cognitive-specific analysis plots

#### MachineFigure/deeplearning_network_viz.py
**Role**: Network visualization for deep learning results  
**Function**: 
- Creates circular network visualizations of task relationships
- Uses t-SNE for dimensionality reduction based on deep learning predictions
- Color-codes nodes by score differences
- Generates publication-ready network diagrams

### 2.4 Advanced Deep Learning Pipeline

#### Deeplearning/Deepscoring_Enhanced.py
**Role**: Two-stage enhanced deep learning with reliability refinement  
**Function**: 
- **Stage 1**: Uses Gemma-3-27b-it for reliability judgment between OPUS and GPT
- **Stage 2**: Trains ReliabilityRefinementModel and FinalScoreModel
- Implements multimodal fusion (text, numeric, deep features)
- Includes attention mechanisms and advanced architectures
- Progress tracking with detailed ETA calculations
- Chunked output for large-scale processing

#### Deeplearning_cognitive/Deepscoring_Enhanced_cognitive.py
**Role**: Enhanced deep learning for cognitive task analysis  
**Function**: 
- Parallel implementation to routine version for cognitive tasks
- Same two-stage architecture with cognitive-specific prompts
- Specialized for cognitive vs. manual task distinctions
- Includes cognitive-specific validation methods

#### gemma3_routine/gemma_processor.py
**Role**: Large-scale task processing using Gemma-3 model  
**Function**: 
- Processes all 23,825 ONET tasks using Gemma-3-27b-it
- Implements automatic recovery and resume capabilities
- A100 GPU optimization with memory management
- Concurrent processing with ThreadPoolExecutor
- Telegram notifications for progress monitoring
- Comprehensive error handling and logging

### 2.5 Statistical Analysis Files

#### rf_threading.py
**Role**: High-performance Random Forest with GPU optimization  
**Function**: 
- CUDA-accelerated version of Random Forest analysis
- PyTorch integration for GPU preprocessing
- Multi-CPU parallel processing for model training
- Enhanced memory management and optimization
- Faster processing for large-scale temporal analysis

### 2.6 Main Analysis File

#### main.do
**Role**: Stata-based econometric analysis and visualization  
**Function**: 
- **Data Integration**: Merges CPS, ONET, and LLM classification results
- **Graph Generation**: Creates publication figures (Graph1.eps, Graph3.pdf, etc.)
- **Econometric Analysis**: Regression analysis with year-occupation-gender interactions
- **Employment Analysis**: Temporal employment share analysis by task categories
- **Validation**: Comparison with existing RTI measures (Autor et al., 2013)
- **Output Generation**: LaTeX tables and EPS figures for publication

## 3. Data Flow Summary

### Input Data
- `Taskvalue.dta`: Raw ONET task descriptions
- `rou.dta`: Routine score data from LLM classifications
- `cog.dta`: Cognitive score data from LLM classifications
- `CPSyearly*.dta`: Current Population Survey data
- Various crosswalk and auxiliary files

### Intermediate Outputs
- Embeddings files (`embeddings_*.pkl`)
- Deep learning results (`deeplearning_results*.pkl`)
- Chunked classification results (`updated_file_*.dta`, `chunk_*.dta`)

### Final Outputs
- Publication graphs (`.eps`, `.pdf` files)
- Regression tables (`.tex` files)
- Enhanced deep learning results (`Deepresult_Enhanced_*.dta`)
- Visualization files (`.png` files)

## 4. Execution Workflow

1. **Configuration**: Set up paths in `config_absolute_path.py`
2. **Initial Classification**: Run LLM classification scripts (OPUS/GPT for routine/cognitive)
3. **Machine Learning**: Execute machine analysis pipeline through `run_analysis.py`
4. **Deep Learning**: Run deep learning predictors and enhanced scoring
5. **Statistical Analysis**: Execute Random Forest analysis with SHAP
6. **Final Analysis**: Run `main.do` for econometric analysis and publication outputs
7. **Maintenance**: Use `clearcache.py` as needed for system cleanup

This integrated pipeline combines cutting-edge LLM capabilities with traditional econometric methods to provide comprehensive analysis of occupational task polarization.

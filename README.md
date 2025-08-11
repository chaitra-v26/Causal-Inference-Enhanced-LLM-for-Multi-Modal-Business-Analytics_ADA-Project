# Causal Inference-Enhanced Large Language Models for Multi-Modal Business Analytics

## 📊 Project Overview

This project combines cutting-edge causal inference techniques with advanced language models to revolutionize business analytics. By analyzing the impact of marketing campaigns on customer spending using robust causal effect estimation techniques, we seamlessly integrate these insights into a language model to generate dynamic, interpretable summaries and actionable recommendations for data-driven business strategies.

## 🎯 Problem Statement

Design a cutting-edge framework combining causal inference and advanced language models to revolutionize business analytics. Analyze the impact of marketing campaigns on customer spending using robust causal effect estimation techniques. Seamlessly integrate these insights into a language model to generate dynamic, interpretable summaries and actionable recommendations, empowering businesses with data-driven strategies for optimized decision-making and measurable success.

## ✨ Key Features & Uniqueness

### 1. **Combination of Causal Inference and LLMs**
- Traditional ML models excel at prediction but fail to provide actionable insights about cause and effect
- This project estimates causal effects (e.g., how marketing campaigns change customer behavior)
- Uses BART language model to summarize effects in natural language format
- Makes complex results interpretable for non-technical stakeholders

### 2. **Cutting-Edge Techniques**
- Leverages state-of-the-art causal modeling techniques (propensity score matching, linear regression)
- Bridges the gap between statistical rigor and user-friendly communication
- Dynamic summarization using BART combines quantitative rigor with qualitative articulation

## 🗂️ Dataset

- **Source**: [Marketing Campaign Dataset - Kaggle](https://www.kaggle.com/datasets/rodsaldanha/arketing-campaign)
- **Size**: 3,554 rows × 21 columns
- **Focus**: Marketing campaign impact on customer spending behavior
- Contains treatment, outcome, and confounder variables for comprehensive causal analysis

### Dataset Description
The dataset provides comprehensive information about marketing campaigns and customer behavior, enabling robust causal inference analysis to understand the effectiveness of different marketing strategies on customer spending patterns.

## 🛠️ Tech Stack

### **Core Technologies**
- **Python** - Primary programming language
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning algorithms

### **Causal Inference**
- **DoWhy** - Causal inference framework
- **statsmodels** - Statistical modeling
- **Propensity Score Matching** - Treatment effect estimation

### **Natural Language Processing**
- **Hugging Face Transformers** - BART model implementation
- **BART (Bidirectional and Auto-Regressive Transformers)** - Text summarization

### **Visualization & Analysis**
- **matplotlib** - Data visualization
- **seaborn** - Statistical data visualization
- **networkx** - Graph analysis for DAGs

## 🚀 Installation & Setup

### Prerequisites
```bash
Python 3.8+
pip or conda package manager
```

### Install Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
pip install dowhy statsmodels
pip install transformers torch
pip install networkx
```

### Alternative using conda
```bash
conda install pandas numpy scikit-learn matplotlib seaborn
conda install -c conda-forge dowhy statsmodels
conda install -c huggingface transformers
pip install torch networkx
```

## 📋 Usage Instructions

### 1. **Clone the Repository**
```bash
git clone https://github.com/your-username/Causal-Inference-Enhanced-LLM-for-Multi-Modal-Business-Analytics.git
cd Causal-Inference-Enhanced-LLM-for-Multi-Modal-Business-Analytics
```

### 2. **Run the Main Notebook**
```bash
# Launch Jupyter Notebook
jupyter notebook ADA_Causal_Inference.ipynb
```

### 3. **Execute the Pipeline**
- Open `ADA_Causal_Inference.ipynb` in Jupyter
- Run all cells sequentially to execute the complete pipeline:
  - Data preprocessing and exploration
  - Causal inference analysis
  - BART-powered report generation

## 📈 Project Workflow

### Phase 1: Data Preparation & Exploration
- [x] Load dataset and check for missing values
- [x] Define treatment, outcome, and confounder variables
- [x] Handle data preprocessing (outliers, normalization, scaling)

### Phase 2: Causal Inference Implementation
- [x] Perform logistic regression for propensity score estimation
- [x] Apply nearest neighbor matching for balanced treatment-control groups
- [x] Compute and visualize Average Treatment Effect (ATE)
- [x] Develop Directed Acyclic Graph (DAG) for causal structure

### Phase 3: Advanced Analysis
- [x] Implement causal model using DoWhy framework
- [x] Conduct effect estimation using matching and regression techniques
- [x] Perform sensitivity analysis to validate causal inference robustness

### Phase 4: LLM Integration
- [x] Integrate Hugging Face's BART model for report generation
- [x] Generate dynamic, interpretable summaries
- [x] Create actionable recommendations

## 📊 Key Results

- **Average Treatment Effect (ATE)**: $220.17
- Successfully implemented end-to-end causal inference pipeline
- Generated automated reports using BART model
- Created comprehensive visualizations for stakeholder communication

## 🔬 Methodology

### Causal Inference Techniques
1. **Propensity Score Matching**: Balances treatment and control groups
2. **Linear Regression**: Estimates causal effects
3. **Sensitivity Analysis**: Validates robustness of findings
4. **DAG Construction**: Represents causal relationships

### LLM Integration
1. **BART Model**: Generates natural language summaries
2. **Dynamic Reporting**: Creates interpretable business insights
3. **Automated Recommendations**: Provides actionable strategies

## 🎓 Learning Outcomes

- In-depth understanding of causal inference techniques for real-world datasets
- Advanced statistical methods implementation (propensity score matching, sensitivity analysis)
- DAG construction and validation for causal relationships
- Machine learning model optimization for treatment effect estimation
- Integration of automated reporting and visualization tools
- Enhanced programming expertise in statistical analysis

## ⚠️ Current Limitations & Future Work

### Unresolved Challenges
1. **Data Quality**: Handling missing/inconsistent data affecting causal calculations
2. **Design Flexibility**: Limited dynamic selection of covariates/treatment variables
3. **Visualization**: Sensitivity analysis needs better visualization for stakeholders
4. **Model Customization**: BART model lacks domain-specific terminology adaptation

### Future Enhancements
- Implement dynamic variable selection interface
- Enhance sensitivity analysis visualization
- Add domain-specific fine-tuning for BART model
- Develop real-time analytics dashboard

## 📚 References

1. Wu, Y., & Wang, W. - "Towards Causal Representation Learning for Multimodal Data"
2. Le, H. D., Xia, X., & Chen, Z. - "Multi-Agent Causal Discovery Using Large Language Models"
3. Smith, J. T., & Chen, L. - "Causal BERT for Explainable Text Analysis in Business Analytics"
4. Zhou, M., & Nair, P. - "Integrating Causal Graphs with Transformer Architectures for Multimodal Data Analysis"
5. Deshpande, S. - "Multi-Modal Causal Inference with Deep Structural Equation Models"
6. Chernozhukov, V., et al. - "Applied Causal Inference Powered by ML and AI"

---

**Course**: UE22AM343AB4 – Advanced Data Analytics  
**Institution**: PES University  

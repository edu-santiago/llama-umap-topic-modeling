# Topic Modelling Applied to Negative Reviews of Elden Ring

A topic modeling project that uses LLaMA 3.2-1B for embedding generation, followed by dimensionality reduction through UMAP and clustering via HDBSCAN to identify common themes in negative reviews of *Elden Ring*. The project demonstrates how contemporary Large Language Model architectures can generate semantically meaningful embeddings that effectively cluster similar issues, enabling the extraction of actionable insights from gaming reviews.

## Overview

User reviews in video games have been integral to digital distribution platforms since their inception, serving as crucial indicators for consumer decision-making by guiding potential players toward or away from specific titles. However, systematic analysis of gaming review topics and their distribution remains understudied in practical applications.

This project presents an approach to Topic Modeling applied to negative reviews of *Elden Ring*, utilizing:
- **LLaMA 3.2-1B** for text embedding generation
- **UMAP** for dimensionality reduction
- **HDBSCAN** for clustering and topic identification
- **Hyperparameter optimization** using multiple evaluation metrics

## Motivation

The video game industry has generated over 400 billion dollars in revenue as of 2024, surpassing the movie, book, and music industries. Unlike passive media, video games rely on user interaction, creating unique experiences for each player.

*Elden Ring* is known for its difficulty and overwhelming challenges compared to other games. Despite winning the Game Awards for Best Game of 2022, *Elden Ring* currently has around 92% positive reviews across all languages and reviews on Steam.

A system that can correctly classify written reviews based on semantic similarity through Topic Modeling could be used to:
- Evaluate common themes among *Elden Ring* players
- Highlight user concerns and track complaint frequencies over time
- Reveal whether technical problem complaints decreased following patch releases
- Provide actionable insights for game developers

## Dataset

The dataset was obtained using Steam's Web API and contains user reviews of *Elden Ring* with:
- Steam IDs
- Playtime in *Elden Ring* (playtime_forever)
- Number of reviews written (num_reviews)
- Number of games owned
- Review sentiment (voted_up: True for positive, False for negative)
- Review text

**Dataset Statistics:**
- Sample size: 5,000 reviews (randomly selected)
- Focus: Negative reviews (voted_up = False)
- Additional analysis: 15,000 reviews for extended clustering

## Installation

### Prerequisites

- Python 3.7+
- CUDA-capable GPU (recommended for LLaMA model)
- pip or conda

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install required packages:
```bash
pip install torch transformers sentence-transformers pandas numpy umap-learn hdbscan scikit-learn bertopic python-dotenv tqdm
```

Or using conda:
```bash
conda install pytorch -c pytorch
pip install transformers sentence-transformers pandas numpy umap-learn hdbscan scikit-learn bertopic python-dotenv tqdm
```

3. Set up environment variables (if using Hugging Face models):
Create a `.env` file in the project root:
```
COLAB_KEY=your_huggingface_token_here
```

## Project Structure

```
.
├── data/
│   └── final_reviews.csv          # Steam reviews dataset
├── models/                         # Saved models and results
│   ├── *_embeddings.npy           # Precomputed embeddings
│   ├── *_hdbscan.pkl              # Trained HDBSCAN models
│   └── *_llama_topics.csv         # Clustering results
├── topic_modelling.ipynb           # Top2Vec baseline approach
├── Topic_Modelling_LLama.ipynb     # Main LLaMA embedding and clustering
├── Topic_Modelling_LLama_analysis.ipynb  # Analysis and evaluation
├── Topic_Modelling_LLama_final.ipynb     # Final optimized clustering
├── Topic_Modelling_BERTopic.ipynb        # BERTopic comparison
├── run_clustering.ipynb            # Clustering execution and results
└── README.md                       # This file
```

## Usage

### Step 1: Data Preparation

The dataset should be in `data/final_reviews.csv` with columns including:
- `review`: Text content of the review
- `voted_up`: Boolean indicating positive (True) or negative (False) review

### Step 2: Generate Embeddings

Open `Topic_Modelling_LLama.ipynb` to:
- Load the LLaMA 3.2-1B model
- Generate embeddings for negative reviews
- Save embeddings for later use

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
model.to(device)

# Generate embeddings
embeddings = get_embeddings(reviews)
```

### Step 3: Hyperparameter Optimization

Open `Topic_Modelling_LLama_analysis.ipynb` or `Topic_Modelling_LLama_final.ipynb` to:
- Optimize UMAP and HDBSCAN hyperparameters
- Evaluate clustering using multiple metrics (Silhouette Score, Calinski-Harabasz Index, Davies-Bouldin Index)
- Find optimal parameter combinations

**UMAP Hyperparameters:**
- n_neighbors: 5, 25, 50
- n_components: 2, 5, 8
- min_dist: 0, 0.1, 0.3, 0.5
- metric: euclidean, cosine

**HDBSCAN Hyperparameters:**
- min_cluster_size: 5, 25, 50
- min_samples: 1, 5, 10
- cluster_selection_epsilon: 0, 0.5, 1
- cluster_selection_method: eom, leaf

### Step 4: Clustering

Open `run_clustering.ipynb` to:
- Load optimized parameters
- Apply UMAP dimensionality reduction
- Perform HDBSCAN clustering
- Analyze and visualize results

### Step 5: Alternative Approaches

**Top2Vec Baseline:**
Open `topic_modelling.ipynb` to explore Top2Vec as a baseline comparison method.

**BERTopic Comparison:**
Open `Topic_Modelling_BERTopic.ipynb` to compare BERTopic results with the LLaMA-based approach.

## Methodology

### Text Embedding

The text embeddings were generated using the Llama 3.2-1B model, run locally on an RTX 3060 GPU with 12 GB of VRAM, 32 GB of RAM, and an AMD Ryzen 5 5600X CPU. The process involved:

1. Passing each negative review through the transformer model
2. Extracting values from the final hidden layer (dimensionality: 512)
3. Using mean pooling over token embeddings to create document-level embeddings

These embeddings capture the semantic similarity of each review's text and serve as the input for subsequent clustering steps.

### Dimensionality Reduction

Before clustering, a dimensionality reduction step was performed using Uniform Manifold Approximation and Projection (UMAP). UMAP was selected for its ability to:
- Preserve the original topological structure of the data
- Aid in forming semantically coherent groups
- Handle high-dimensional embedding spaces effectively

### Clustering

The clustering was conducted using Hierarchical Density-Based Spatial Clustering of Applications with Noise (HDBSCAN). This method was chosen because it:
- Generates hierarchical clusters that can be fine-tuned
- Identifies core, frontier, and outlier points
- Discards reviews that do not fit into meaningful groups
- Ensures only relevant and well-defined clusters are retained

### Evaluation Metrics

Three metrics were employed for hyperparameter optimization:

1. **Silhouette Score**: Evaluates the distinctiveness of each cluster by measuring similarity within clusters compared to other clusters
2. **Calinski-Harabasz Index**: Measures the ratio of intra-cluster dispersion to inter-cluster separation
3. **Davies-Bouldin Index**: Captures the structural integrity of clusters relative to each other

Since these metrics operate on different scales, their values were normalized to a range between 0 and 1. The optimization process aimed to maximize the mean value across all three metrics.

## Results

### Initial Clustering

The topic modeling analysis was conducted on a randomly selected sample of 5,000 reviews, resulting in three distinct topic groups, along with an additional outlier group:

| Topic | Number of Reviews |
|-------|-------------------|
| Technical Problems | 1,451 |
| Constructive Criticisms | 1,367 |
| Non-constructive Criticisms | 906 |
| Outliers | 1,276 |

The clustering achieved a **Silhouette Score of 0.38**, indicating moderate separation between the identified clusters.

### Topic Descriptions

**Technical Problems:**
- Crashes and performance issues
- Save file problems
- Controller/input issues
- Multiplayer connectivity problems
- PC optimization concerns

**Constructive Criticisms:**
- Game design feedback
- Content repetition concerns
- Story and narrative critiques
- Difficulty scaling issues
- Open world structure feedback

**Non-constructive Criticisms:**
- Simple negative statements without detail
- Comparisons to other games without context
- Emotional reactions without specific issues

### Hierarchical Clustering

Subsequently, the same model and optimization process was applied exclusively to the Constructive Criticism cluster, yielding a silhouette coefficient of 0.3 and revealing two distinct groups:

| Topic | Number of Reviews |
|-------|-------------------|
| Difficulty | 559 |
| Game Scope | 236 |
| Outliers | 572 |

**Difficulty Cluster:**
- Learning curve concerns
- Boss difficulty complaints
- Need for external guides
- Time investment requirements

**Game Scope Cluster:**
- Open world size and pacing
- Content repetition
- Exploration vs. progression balance
- World structure and layout

## Key Findings

1. **Technical issues are prominent**: Nearly 30% of negative reviews focus on technical problems, suggesting potential optimization issues, especially for PC players with diverse hardware configurations.

2. **Constructive feedback is actionable**: The constructive criticism cluster (27% of reviews) provides specific, actionable feedback that could inform game development decisions.

3. **Difficulty is a major concern**: When analyzing constructive criticisms, difficulty-related concerns represent the largest subgroup, indicating that while difficulty is part of the game's design philosophy, it remains a significant barrier for many players.

4. **Open world structure challenges**: The game scope cluster reveals concerns about the expansive open-world structure, with players reporting spending disproportionate time traversing the environment rather than engaging with core gameplay mechanics.

5. **LLaMA embeddings are effective**: The use of LLaMA 3.2-1B for embedding generation successfully captured semantic relationships, enabling meaningful topic identification despite the diverse nature of gaming reviews.

## Notebooks

- **topic_modelling.ipynb**: Top2Vec baseline approach for comparison
- **Topic_Modelling_LLama.ipynb**: Main notebook for LLaMA embedding generation and initial clustering
- **Topic_Modelling_LLama_analysis.ipynb**: Hyperparameter optimization and evaluation
- **Topic_Modelling_LLama_final.ipynb**: Final optimized clustering with best parameters
- **Topic_Modelling_BERTopic.ipynb**: BERTopic comparison approach
- **run_clustering.ipynb**: Clustering execution and result analysis

## Future Work

1. **Alternative LLM architectures**: Explore other Large Language Model architectures for embedding generation, potentially with larger models for improved semantic understanding

2. **Granular subtopic decomposition**: Decompose primary topics into more granular subtopics to identify specific common issues

3. **Linked prompting strategies**: Investigate the effectiveness of linked prompting strategies where topic modeling generates common issues and subsequently categorizes individual reviews

4. **Temporal analysis**: Track complaint frequencies over time to evaluate the impact of patches and updates

5. **Multi-language support**: Extend the analysis to reviews in multiple languages

6. **Positive review analysis**: Apply the same methodology to positive reviews to identify what players appreciate most

## References

This project is based on research presented in:
> **Topic Modelling Applied to Negative Reviews of Elden Ring**
> 
> Eduardo N. S. Ramos  
> Departamento de Engenharia Elétrica, PUC-RJ

For the full academic paper, see `Template_SBC/template-latex/sbc-template.tex`

## Author

**Eduardo N. S. Ramos**  
Email: eduardonsantiago@aluno.puc-rio.br

## License

This project is for academic/research purposes.

## Acknowledgments

- Meta for the LLaMA 3.2-1B model
- Steam Web API for providing review data
- UMAP and HDBSCAN communities for clustering tools
- Hugging Face for transformer model infrastructure

---

**Note**: The implementation of Topic Modeling utilizing LLaMA coupled with clustering algorithms yielded effective results in identifying common thematic patterns among *Elden Ring* players' feedback. The initial clustering analysis revealed that a substantial portion of players encountered legitimate concerns regarding the game's difficulty and scope, alongside technical issues. The methodology demonstrates that contemporary Large Language Model architectures can generate semantically meaningful embeddings that effectively cluster similar issues, enabling the extraction of actionable insights from negative reviews.


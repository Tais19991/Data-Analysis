## Clustering Bank Clients for Targeted Marketing Strategies

### Overview
This project applies advanced clustering techniques to segment bank clients based on their spending and credit behavior. The segmentation enables personalized marketing strategies and better customer engagement by identifying unique financial personas.

### Key Features
- Problem: Identifying distinct customer profiles using credit card usage data.
- Data Analysis: Comprehensive exploration of customer features, including balance, purchase behavior, and credit limits.
- Clustering: Utilizes K-means Clustering and Dimensionality Reduction (PCA and Autoencoders) to group clients into meaningful segments.
- Insights: Detailed cluster analysis reveals distinct spending patterns, enabling actionable marketing strategies.


### Methodology

1.Data Preprocessing:

- Handled missing values and outliers.
- Scaled data using StandardScaler.

2.Exploratory Data Analysis:

- Analyzed correlations and distributions to understand customer behavior.
- Identified patterns in credit usage and spending.

3. Clustering:

- Used the Elbow Method to determine optimal clusters.
- Created initial 8 clusters, refined into 4 broader categories using PCA.

4. Dimensionality Reduction:
   
- Applied PCA and Autoencoders for visualization and improved cluster separation.

5.Visualization:

- Generated heatmaps, scatter plots, and cluster center comparisons for clear insights.


### Findings
**Key Clusters:**
- Cluster 0: Low spenders with high credit limits and high cash advance usage (Cash Advance Enthusiasts)
- Cluster 1: Low spenders with low balances (Frugal Users)
- Cluster 2: High spenders with moderate balances (Balanced Spenders)
- Cluster 3: High spenders with high credit and balances (Luxury Seekers)
 
**Dominant Group:** Cluster 1 accounts for ~70% of clients, indicating potential for retention strategies targeting low-spend segments.

<img src="/assets/Bank_clients_data.gif" alt="Alternate text" width="500"/>



### Recommendations
- Personalized Offers: Tailor credit and reward programs to each cluster's needs.
- Engagement Strategies: Enhance services for low-engagement groups (Cluster 1) to increase activity.
- Premium Services: Develop exclusive products for high-balance, high-spend clusters (Clusters 2 and 3).

### Tools and Libraries

- Programming: Python
- Libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, TensorFlow/Keras

### How to Use
Clone the repository.
Run the Jupyter notebook to explore and customize the analysis.

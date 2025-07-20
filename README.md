Customer Segmentation and Personalized Marketing (Nespresso Case Study)
Project Overview
This project demonstrates how data science techniques can be used to segment retail customers and develop personalized marketing strategies, based on transactional data from Nespresso. Using Python, the dataset was cleaned and prepared, k-means clustering was applied to identify key customer groups, and PCA was utilized for effective visualization. The main goal was to recommend targeted marketing actions to boost customer engagement and retention.

File Structure
/Market-Segmentation-Project/
├── Market_Segmentation.py        # Python script with analysis and clustering
├── Market_Segmentation.xlsx      # Excel file with anonymized dataset
├── plots/                        # Visualizations (PCA and cluster plots)
│   ├── cluster_plot.png
│   └── pca_plot.png
├── README.md                     # Project documentation
└── summary_report.pdf            # (Optional) Executive summary/report

Technologies Used
Python 3.x

Pandas (data manipulation and cleaning)

NumPy (numerical computations)

scikit-learn (machine learning: k-means clustering, PCA)

Matplotlib (data visualization)

Seaborn (advanced data visualizations)

Microsoft Excel (data storage and initial preparation)

Features
Data cleaning and preprocessing using Pandas

Customer segmentation via k-means clustering (scikit-learn)

Visualization of customer segments with PCA and Matplotlib

Detailed profiling of each segment

Business recommendations for targeted marketing

Visualizations (Python)
The following charts were generated using Matplotlib and Seaborn based on the analyzed retail data.
Elbow Plot
<img width="521" height="325" alt="image" src="https://github.com/user-attachments/assets/4a63ac2c-b00b-48c8-9c91-f4fb048ec7e2" />

Silhouette Score Chart
<img width="412" height="257" alt="image" src="https://github.com/user-attachments/assets/9c0ef7ff-47c1-4e38-9b2f-c846f8c0be67" />

Dendrogram – Hierarchical Clustering
<img width="380" height="266" alt="image" src="https://github.com/user-attachments/assets/9f21b436-8be5-4211-84fb-fce32383c1ee" />

Customer Segment Separability (PCA)
<img width="401" height="302" alt="image" src="https://github.com/user-attachments/assets/1d92e0c9-a5e0-4bd1-9aa4-9756ffd38aca" />

Comparison of Average Basket Value Between Segments
<img width="446" height="279" alt="image" src="https://github.com/user-attachments/assets/6dfc8b5f-3389-46c0-8069-17343b15b2be" />

Percentage Distribution of Gender by Customer Segments
<img width="382" height="287" alt="image" src="https://github.com/user-attachments/assets/a31f15a5-265c-44e9-85b4-afaf18c938d4" />

Average Proportion of Binary Variables by Customer Segment
<img width="416" height="243" alt="image" src="https://github.com/user-attachments/assets/34b50e30-7745-47c2-9624-2e82f87a28a1" />

Setup Instructions
Clone the repository:

git clone https://github.com/yourusername/Market-Segmentation-Project.git
cd Market-Segmentation-Project

Install required packages:

pip install pandas numpy scikit-learn matplotlib seaborn openpyxl

Run the analysis script:

python Market_Segmentation.py

Author
Karina @karinastukan


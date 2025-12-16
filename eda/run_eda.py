"""
Exploratory Data Analysis (EDA) Script

This script performs comprehensive EDA on the student dropout dataset:
- Univariate analysis (histograms, KDE, outliers)
- Bivariate analysis (feature vs dropout)
- Correlation analysis
- Class balance analysis

Author: AI Project
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from scipy import stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def detect_outliers_iqr(df, column):
    """
    Detect outliers using IQR method.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    column : str
        Column name to analyze
    
    Returns:
    --------
    dict
        Dictionary with outlier statistics
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    return {
        'column': column,
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'outlier_count': len(outliers),
        'outlier_percentage': (len(outliers) / len(df)) * 100
    }


def univariate_analysis(df, output_dir='eda/plots'):
    """
    Perform univariate analysis on all features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    output_dir : str
        Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    features = ['attendance_percentage', 'academic_score', 
                'indiscipline_count', 'online_engagement_score']
    
    outlier_results = []
    
    logger.info("Performing univariate analysis...")
    
    for feature in features:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Univariate Analysis: {feature}', fontsize=16, fontweight='bold')
        
        # Histogram
        axes[0, 0].hist(df[feature], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        axes[0, 0].set_title(f'Histogram of {feature}')
        axes[0, 0].set_xlabel(feature)
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # KDE Plot
        df[feature].plot(kind='density', ax=axes[0, 1], color='coral', linewidth=2)
        axes[0, 1].set_title(f'KDE Plot of {feature}')
        axes[0, 1].set_xlabel(feature)
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Box Plot
        axes[1, 0].boxplot(df[feature], vert=True)
        axes[1, 0].set_title(f'Box Plot of {feature}')
        axes[1, 0].set_ylabel(feature)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Statistics
        stats_text = f"""
        Mean: {df[feature].mean():.2f}
        Median: {df[feature].median():.2f}
        Std: {df[feature].std():.2f}
        Min: {df[feature].min():.2f}
        Max: {df[feature].max():.2f}
        Skewness: {df[feature].skew():.2f}
        Kurtosis: {df[feature].kurtosis():.2f}
        """
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, 
                        verticalalignment='center', family='monospace',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Statistics')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/univariate_{feature}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Detect outliers
        outlier_info = detect_outliers_iqr(df, feature)
        outlier_results.append(outlier_info)
        
        logger.info(f"  ✓ {feature}: {outlier_info['outlier_count']} outliers ({outlier_info['outlier_percentage']:.2f}%)")
    
    return outlier_results


def bivariate_analysis(df, output_dir='eda/plots'):
    """
    Perform bivariate analysis (features vs dropout).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    output_dir : str
        Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    features = ['attendance_percentage', 'academic_score', 
                'indiscipline_count', 'online_engagement_score']
    
    logger.info("Performing bivariate analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Bivariate Analysis: Features vs Dropout', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    for idx, feature in enumerate(features):
        # Violin plot
        sns.violinplot(data=df, x='dropout', y=feature, ax=axes[idx], palette='Set2')
        axes[idx].set_title(f'{feature} vs Dropout')
        axes[idx].set_xlabel('Dropout (0=No, 1=Yes)')
        axes[idx].set_ylabel(feature)
        axes[idx].grid(True, alpha=0.3)
        
        # Add mean lines
        for dropout_val in [0, 1]:
            mean_val = df[df['dropout'] == dropout_val][feature].mean()
            axes[idx].axhline(mean_val, color='red', linestyle='--', 
                            alpha=0.7, linewidth=2, label=f'Mean (Dropout={dropout_val})')
        
        if idx == 0:
            axes[idx].legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/bivariate_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Individual detailed plots
    for feature in features:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'{feature} vs Dropout - Detailed Analysis', fontsize=14, fontweight='bold')
        
        # Box plot
        sns.boxplot(data=df, x='dropout', y=feature, ax=axes[0], palette='Set2')
        axes[0].set_title(f'Box Plot: {feature} by Dropout')
        axes[0].set_xlabel('Dropout (0=No, 1=Yes)')
        axes[0].set_ylabel(feature)
        axes[0].grid(True, alpha=0.3)
        
        # Distribution comparison
        df[df['dropout'] == 0][feature].plot(kind='density', ax=axes[1], 
                                            label='No Dropout', color='green', linewidth=2)
        df[df['dropout'] == 1][feature].plot(kind='density', ax=axes[1], 
                                            label='Dropout', color='red', linewidth=2)
        axes[1].set_title(f'Distribution Comparison: {feature}')
        axes[1].set_xlabel(feature)
        axes[1].set_ylabel('Density')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/bivariate_{feature}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info("  ✓ Bivariate analysis plots saved")


def correlation_analysis(df, output_dir='eda/plots'):
    """
    Perform correlation analysis.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    output_dir : str
        Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Performing correlation analysis...")
    
    # Calculate correlation matrix
    corr_matrix = df.corr()
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix Heatmap', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("  ✓ Correlation heatmap saved")
    
    return corr_matrix


def class_balance_analysis(df):
    """
    Analyze class balance in the target variable.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    
    Returns:
    --------
    dict
        Class balance statistics
    """
    logger.info("Analyzing class balance...")
    
    dropout_counts = df['dropout'].value_counts()
    dropout_percentages = df['dropout'].value_counts(normalize=True) * 100
    
    balance_info = {
        'class_0_count': int(dropout_counts[0]),
        'class_1_count': int(dropout_counts[1]),
        'class_0_percentage': float(dropout_percentages[0]),
        'class_1_percentage': float(dropout_percentages[1]),
        'imbalance_ratio': float(dropout_counts[0] / dropout_counts[1]),
        'is_balanced': abs(dropout_percentages[0] - dropout_percentages[1]) < 10
    }
    
    # Visualize class balance
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Count plot
    dropout_counts.plot(kind='bar', ax=axes[0], color=['green', 'red'], alpha=0.7)
    axes[0].set_title('Class Distribution (Count)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Dropout (0=No, 1=Yes)')
    axes[0].set_ylabel('Count')
    axes[0].set_xticklabels(['No Dropout', 'Dropout'], rotation=0)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add count labels
    for i, v in enumerate(dropout_counts):
        axes[0].text(i, v + 50, str(v), ha='center', va='bottom', fontweight='bold')
    
    # Percentage pie chart
    axes[1].pie(dropout_percentages, labels=['No Dropout', 'Dropout'], 
               autopct='%1.1f%%', startangle=90, colors=['green', 'red'], 
               explode=(0.05, 0.05), shadow=True)
    axes[1].set_title('Class Distribution (Percentage)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('eda/plots/class_balance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  ✓ Class 0 (No Dropout): {balance_info['class_0_count']} ({balance_info['class_0_percentage']:.2f}%)")
    logger.info(f"  ✓ Class 1 (Dropout): {balance_info['class_1_count']} ({balance_info['class_1_percentage']:.2f}%)")
    logger.info(f"  ✓ Imbalance Ratio: {balance_info['imbalance_ratio']:.2f}")
    logger.info(f"  ✓ Is Balanced: {balance_info['is_balanced']}")
    
    return balance_info


def generate_eda_report(df, outlier_results, corr_matrix, balance_info, output_file='eda/eda_report.md'):
    """
    Generate comprehensive EDA report in Markdown format.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    outlier_results : list
        List of outlier analysis results
    corr_matrix : pd.DataFrame
        Correlation matrix
    balance_info : dict
        Class balance information
    output_file : str
        Path to save the report
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    logger.info("Generating EDA report...")
    
    report = f"""# Exploratory Data Analysis Report
## Student Dropout Prediction Dataset

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. Dataset Overview

- **Total Records**: {len(df):,}
- **Total Features**: {len(df.columns) - 1} (excluding target)
- **Target Variable**: `dropout` (Binary: 0 = No Dropout, 1 = Dropout)

### Features:
1. `attendance_percentage` - Student attendance percentage (0-100)
2. `academic_score` - Academic performance score (0-10)
3. `indiscipline_count` - Number of indiscipline incidents (0-10)
4. `online_engagement_score` - Online engagement score (0-100)
5. `dropout` - Target variable (0/1)

---

## 2. Descriptive Statistics

### Summary Statistics:
```
{df.describe().to_string()}
```

---

## 3. Univariate Analysis

### Feature Distributions

#### Attendance Percentage
- **Mean**: {df['attendance_percentage'].mean():.2f}%
- **Median**: {df['attendance_percentage'].median():.2f}%
- **Std Dev**: {df['attendance_percentage'].std():.2f}
- **Range**: {df['attendance_percentage'].min():.2f}% - {df['attendance_percentage'].max():.2f}%

#### Academic Score
- **Mean**: {df['academic_score'].mean():.2f}/10
- **Median**: {df['academic_score'].median():.2f}/10
- **Std Dev**: {df['academic_score'].std():.2f}
- **Range**: {df['academic_score'].min():.2f} - {df['academic_score'].max():.2f}

#### Indiscipline Count
- **Mean**: {df['indiscipline_count'].mean():.2f}
- **Median**: {df['indiscipline_count'].median():.2f}
- **Std Dev**: {df['indiscipline_count'].std():.2f}
- **Range**: {df['indiscipline_count'].min():.2f} - {df['indiscipline_count'].max():.2f}

#### Online Engagement Score
- **Mean**: {df['online_engagement_score'].mean():.2f}%
- **Median**: {df['online_engagement_score'].median():.2f}%
- **Std Dev**: {df['online_engagement_score'].std():.2f}
- **Range**: {df['online_engagement_score'].min():.2f}% - {df['online_engagement_score'].max():.2f}%

### Outlier Detection (IQR Method)

"""
    
    for outlier_info in outlier_results:
        report += f"""
#### {outlier_info['column']}
- **Q1**: {outlier_info['Q1']:.2f}
- **Q3**: {outlier_info['Q3']:.2f}
- **IQR**: {outlier_info['IQR']:.2f}
- **Lower Bound**: {outlier_info['lower_bound']:.2f}
- **Upper Bound**: {outlier_info['upper_bound']:.2f}
- **Outliers Detected**: {outlier_info['outlier_count']} ({outlier_info['outlier_percentage']:.2f}%)
"""
    
    report += f"""
---

## 4. Bivariate Analysis

### Feature vs Dropout Relationships

#### Attendance Percentage vs Dropout
- **Mean (No Dropout)**: {df[df['dropout'] == 0]['attendance_percentage'].mean():.2f}%
- **Mean (Dropout)**: {df[df['dropout'] == 1]['attendance_percentage'].mean():.2f}%
- **Difference**: {df[df['dropout'] == 0]['attendance_percentage'].mean() - df[df['dropout'] == 1]['attendance_percentage'].mean():.2f}%

#### Academic Score vs Dropout
- **Mean (No Dropout)**: {df[df['dropout'] == 0]['academic_score'].mean():.2f}/10
- **Mean (Dropout)**: {df[df['dropout'] == 1]['academic_score'].mean():.2f}/10
- **Difference**: {df[df['dropout'] == 0]['academic_score'].mean() - df[df['dropout'] == 1]['academic_score'].mean():.2f}

#### Indiscipline Count vs Dropout
- **Mean (No Dropout)**: {df[df['dropout'] == 0]['indiscipline_count'].mean():.2f}
- **Mean (Dropout)**: {df[df['dropout'] == 1]['indiscipline_count'].mean():.2f}
- **Difference**: {df[df['dropout'] == 1]['indiscipline_count'].mean() - df[df['dropout'] == 0]['indiscipline_count'].mean():.2f}

#### Online Engagement Score vs Dropout
- **Mean (No Dropout)**: {df[df['dropout'] == 0]['online_engagement_score'].mean():.2f}%
- **Mean (Dropout)**: {df[df['dropout'] == 1]['online_engagement_score'].mean():.2f}%
- **Difference**: {df[df['dropout'] == 0]['online_engagement_score'].mean() - df[df['dropout'] == 1]['online_engagement_score'].mean():.2f}%

**Insights:**
- Lower attendance is associated with higher dropout rates
- Lower academic scores correlate with dropout
- Higher indiscipline counts are linked to dropout
- Lower online engagement is associated with dropout

---

## 5. Correlation Analysis

### Correlation Matrix:
```
{corr_matrix.to_string()}
```

### Key Correlations with Dropout:
- **Attendance vs Dropout**: {corr_matrix.loc['attendance_percentage', 'dropout']:.3f}
- **Academic Score vs Dropout**: {corr_matrix.loc['academic_score', 'dropout']:.3f}
- **Indiscipline vs Dropout**: {corr_matrix.loc['indiscipline_count', 'dropout']:.3f}
- **Engagement vs Dropout**: {corr_matrix.loc['online_engagement_score', 'dropout']:.3f}

**Interpretation:**
- Negative correlations indicate that lower values of the feature are associated with dropout
- Positive correlations indicate that higher values of the feature are associated with dropout

---

## 6. Class Balance Analysis

### Distribution:
- **Class 0 (No Dropout)**: {balance_info['class_0_count']:,} ({balance_info['class_0_percentage']:.2f}%)
- **Class 1 (Dropout)**: {balance_info['class_1_count']:,} ({balance_info['class_1_percentage']:.2f}%)
- **Imbalance Ratio**: {balance_info['imbalance_ratio']:.2f}:1

### Balance Status:
- **Is Balanced**: {'Yes' if balance_info['is_balanced'] else 'No'}

### Recommendation:
"""
    
    if not balance_info['is_balanced']:
        report += """
**⚠️ Class Imbalance Detected!**

The dataset shows class imbalance. It is recommended to:
1. Apply SMOTE (Synthetic Minority Oversampling Technique) during preprocessing
2. Use class weights in the model training
3. Consider stratified sampling for train-test split
4. Use appropriate evaluation metrics (F1-score, ROC-AUC) instead of just accuracy
"""
    else:
        report += """
**✓ Classes are relatively balanced.**

Standard train-test split and evaluation metrics can be used.
"""
    
    report += f"""
---

## 7. Visualizations

All plots have been saved to `eda/plots/` directory:

### Univariate Analysis:
- `univariate_attendance_percentage.png`
- `univariate_academic_score.png`
- `univariate_indiscipline_count.png`
- `univariate_online_engagement_score.png`

### Bivariate Analysis:
- `bivariate_analysis.png` (Overview)
- `bivariate_attendance_percentage.png`
- `bivariate_academic_score.png`
- `bivariate_indiscipline_count.png`
- `bivariate_online_engagement_score.png`

### Correlation:
- `correlation_heatmap.png`

### Class Balance:
- `class_balance.png`

---

## 8. Key Insights

1. **Attendance is a strong predictor**: Students with lower attendance have significantly higher dropout rates.

2. **Academic performance matters**: Lower academic scores are strongly correlated with dropout.

3. **Discipline issues**: Higher indiscipline counts are associated with dropout.

4. **Engagement is important**: Lower online engagement scores correlate with dropout.

5. **Feature relationships**: Features show expected correlations with the target variable.

6. **Data quality**: The dataset has reasonable distributions with some outliers that may need handling.

---

## 9. Recommendations for Modeling

1. **Preprocessing**:
   - Handle outliers (consider capping or transformation)
   - Apply feature scaling (StandardScaler)
   - Apply SMOTE if class imbalance is significant

2. **Feature Engineering**:
   - Consider creating interaction features
   - May create bins for categorical-like features

3. **Model Selection**:
   - Try tree-based models (RandomForest, XGBoost) as they handle non-linear relationships well
   - Consider ensemble methods for better performance

4. **Evaluation**:
   - Use multiple metrics: Accuracy, F1-score, ROC-AUC, Precision, Recall
   - Use stratified cross-validation
   - Pay attention to confusion matrix

5. **Feature Importance**:
   - Analyze feature importance after training
   - Identify most predictive features

---

*End of Report*
"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"  ✓ EDA report saved to {output_file}")


def main():
    """Main function to run complete EDA."""
    logger.info("="*60)
    logger.info("STARTING EXPLORATORY DATA ANALYSIS")
    logger.info("="*60)
    
    # Load dataset
    dataset_path = 'data/dataset.csv'
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found at {dataset_path}")
        logger.error("Please run data_generator.py first to create the dataset.")
        return
    
    df = pd.read_csv(dataset_path)
    logger.info(f"Dataset loaded: {df.shape}")
    
    # Create output directory
    os.makedirs('eda/plots', exist_ok=True)
    
    # Run analyses
    outlier_results = univariate_analysis(df)
    bivariate_analysis(df)
    corr_matrix = correlation_analysis(df)
    balance_info = class_balance_analysis(df)
    
    # Generate report
    generate_eda_report(df, outlier_results, corr_matrix, balance_info)
    
    logger.info("="*60)
    logger.info("EDA COMPLETED SUCCESSFULLY!")
    logger.info("="*60)
    logger.info(f"📊 Plots saved to: eda/plots/")
    logger.info(f"📄 Report saved to: eda/eda_report.md")


if __name__ == "__main__":
    main()

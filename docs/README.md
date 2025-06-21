# Python in Excel Use Cases Documentation

Welcome to the documentation for Python in Excel Analytics! This guide provides detailed examples and templates for leveraging Python's analytical capabilities directly within Excel.

## 🚀 Getting Started with Python in Excel

Before diving into the use cases, ensure your Python environment is properly configured:

1. Open Excel and enter this command in a new cell to check available libraries:
   ```python
   =PY(
        import sys
        import subprocess
        import pandas as pd
        # Run pip list command and capture the output
        result = subprocess.run([sys.executable, "-m", "pip", "list"], capture_output=True, text=True).stdout
        # Split the output into lines and skip the header lines
        lines = result.strip().split('\n')[2:]
        # Parse each line into package name and version
        packages = [line.split() for line in lines]
        # Create a DataFrame with two columns: Package and Version
        df = pd.DataFrame(packages, columns=["Package", "Version"])
   )
   ```
2. Execute with Ctrl+Enter
3. Save the output to `piplist.txt` for reference
4. Verify core libraries are installed (pandas, numpy, matplotlib, seaborn, statsmodels)

## 📚 Use Case Categories

### 01 - Financial Analysis
- [Portfolio Optimization](./01-Financial%20Analysis_01-Portfolio%20Optimization.md)
- [Financial Statement Analysis](./01-Financial%20Analysis_02-Financial%20Statement%20Analysis.md)
- [Investment Analysis](./01-Financial%20Analysis_03-Investment%20Analysis.md)
- [Risk Analysis with VaR](./01-Financial%20Analysis_04-Risk%20Analysis%20with%20VaR%20(Value%20at%20Risk).md)

### 02 - Business Intelligence
- [Sales Analytics](./02-Business%20Intelligence_01-Sales%20Analytics.md)
- [Marketing Analysis](./02-Business%20Intelligence_02-Marketing%20Analysis.md)
- [Customer Analytics](./02-Business%20Intelligence_03-Customer%20Analytics.md)

### 03 - Data Cleaning & Preparation
- [Data Quality Assessment](./03-Data%20Cleaning%20%26%20Preparation_01-Data%20Quality%20Assessment.md)
- [Data Transformation](./03-Data%20Cleaning%20%26%20Preparation_02-Data%20Transformation.md)
- [Data Integration](./03-Data%20Cleaning%20%26%20Preparation_03-Data%20Integration.md)
- [Data Reshaping](./03-Data%20Cleaning%20%26%20Preparation_04-Data%20Reshaping.md)

### 04 - Statistical Analysis
- [Descriptive Statistics](./04-Statistical%20Analysis_01-Descriptive%20Statistics.md)
- [Inferential Statistics](./04-Statistical%20Analysis_02-Inferential%20Statistics.md)
- [Time Series Analysis](./04-Statistical%20Analysis_03-Time%20Series%20Analysis.md)

### 05 - Predictive Modeling
- [Regression (Predicting Continuous Values)](./05-Predictive%20Modeling_01-Regression%20(Predicting%20Continuous%20Values).md)
- [Classification (Predicting Categorical Values)](./05-Predictive%20Modeling_02-Classification%20(Predicting%20Categorical%20Values).md)
- [Time Series Forecasting](./05-Predictive%20Modeling_03-Time%20Series%20Forecasting.md)

### 06 - Visualization
- [Basic Plots (Line, Bar, Scatter)](./06-Visualization_01-Basic%20Plots%20(Line,Bar,Scatter).md)
- [Distribution Plots](./06-Visualization_02-Distribution%20Plots%20(Histogram,%20Box%20Plot,%20KDE).md)
- [Relationship Plots](./06-Visualization_03-Relationship%20Plots%20(Scatter,%20Pair%20Plot,%20Heatmap).md)
- [Composition Plots](./06-Visualization_04-Composition%20Plots%20(Pie,%20Stacked%20Bar)..md)
- [Geospatial Plots](./06-Visualization_05-Geospatial%20Plots..md)

### 07 - Reporting & Automation
- [Generating Summaries](./07-Reporting%20%26%20Automation_01-Generating%20Summaries.md)
- [Generating Reports](./07-Reporting%20%26%20Automation_02-Generating%20Reporting.md)
- [Parameterization](./07-Reporting%20%26%20Automation_03-Parameterization.md)
- [Conditional Formatting](./07-Reporting%20%26%20Automation_04-Conditional%20Formatting.md)
- [User-Defined Functions (UDFs)](./07-Reporting%20%26%20Automation_05-User-Defined%20Functions%20(UDFs).md)

## 📋 How to Use These Guides

Each use case document follows a consistent structure:
1. **Overview** - Problem description and business context
2. **Prerequisites** - Required Python libraries and Excel setup
3. **Step-by-Step Guide** - Detailed implementation instructions
4. **Code Examples** - Ready-to-use Python code snippets
5. **Tips & Best Practices** - Optimization and troubleshooting advice

## 🔧 Setup Requirements

Core Python libraries required for most use cases:
- pandas
- numpy
- matplotlib
- seaborn
- statsmodels

Additional libraries may be required for specific use cases. Check the `piplist.txt` output from your environment setup to verify available packages.

## 🤝 Contributing

We welcome contributions! For major changes:
1. Open an issue first to discuss your proposal
2. Follow the existing documentation structure
3. Include clear examples and explanations
4. Test all code examples in Excel

## 📄 License

This project is licensed under The Unlicense. See https://unlicense.org/ for details.

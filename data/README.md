# Public Datasets for PythonInExcel

This is a curated list of free public datasets that you can use with PythonInExcel for data analysis, machine learning, and visualization projects.

## General Dataset Collections

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php) - The most comprehensive collection of machine learning datasets
- [Awesome Public Datasets](https://github.com/awesomedata/awesome-public-datasets) - A curated list of high-quality open datasets
- [Google Dataset Search](https://datasetsearch.research.google.com/) - Search engine for datasets
- [Kaggle Datasets](https://www.kaggle.com/datasets) - Large collection of datasets with active community

## Financial Data

- [Yahoo Finance](https://finance.yahoo.com/) - Free stock market data
- [World Bank Open Data](https://data.worldbank.org/) - Global development data
- [Federal Reserve Economic Data (FRED)](https://fred.stlouisfed.org/) - US economic data
- [Quandl](https://www.quandl.com/tools/python) - Financial and economic data (some free datasets)

## Business & Marketing

- [Google Trends](https://trends.google.com/trends/) - Search trends data
- [Yelp Open Dataset](https://www.yelp.com/dataset) - Business reviews and ratings
- [Amazon Product Reviews](https://s3.amazonaws.com/amazon-reviews-pds/readme.html) - Product reviews dataset

## Scientific & Research

- [NASA Open Data](https://data.nasa.gov/) - Space and earth science data
- [NOAA Climate Data](https://www.ncdc.noaa.gov/cdo-web/) - Weather and climate data
- [NIH Open Data](https://datascience.nih.gov/data-sharing) - Health and medical research data

## Machine Learning Practice Datasets

1. **Classification**
   - [Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris) - Classic flower classification
   - [Adult Income Dataset](https://github.com/jbrownlee/Datasets/blob/master/adult-all.csv) - Income prediction based on census data
   - [Wine Quality](https://archive.ics.uci.edu/ml/datasets/wine+quality) - Wine quality classification

2. **Regression**
   - [Boston Housing](https://www.kaggle.com/datasets/schirmerchad/bostonhousingsmlr) - House price prediction
   - [Bike Sharing](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset) - Bicycle rental prediction
   - [Energy Consumption](https://archive.ics.uci.edu/ml/datasets/Energy+efficiency) - Building energy efficiency

3. **Time Series**
   - [Air Quality](https://archive.ics.uci.edu/ml/datasets/Air+Quality) - Air quality measurements
   - [Stock Prices](https://www.alphavantage.co/) - Real-time and historical stock data
   - [Weather History](https://www.kaggle.com/datasets/muthuj7/weather-dataset) - Historical weather data

## Using These Datasets

1. Most datasets can be loaded directly using pandas:
```python
import pandas as pd
url = "dataset_url_here"
df = pd.read_csv(url)
```

2. For datasets requiring authentication or API keys, refer to their documentation for proper access methods.

3. Always check the dataset's license and terms of use before implementing in production environments.

## Contributing

Feel free to suggest additional datasets by opening a pull request!

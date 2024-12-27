NYC Traffic and Collision Analysis
This project analyzes traffic and collision data in New York City using various statistical methods and machine learning models. The analysis focuses on the relationship between traffic incidents and weather conditions, as well as predicting traffic volumes.

File Descriptions

Python Scripts
CrashesByPrecipitation.py: Executes a statistical analysis to explore the relationship between crashes and precipitation levels.

CrashesByWeather.py: Executes a statistical analysis to explore the relationship between crashes and various weather conditions.

CreateNYCCollisionHeatMap.py: Generates an HTML document containing a heatmap visualizing collision data across New York City.

CreateNYCTrafficHeatMap.py: Generates an HTML document containing a heatmap visualizing traffic data across New York City.

ML Training.py: Trains a machine learning model using the available datasets.

ML Executing.py: Executes the trained machine learning model to make predictions.

ML2.py: Utilizes a RandomForestRegressor machine learning model to predict traffic volumes specifically for weekends.

scatter1.py: Generates a scatterplot visualizing traffic volumes across New York City.

NodeWay.py: Calculates and displays the least trafficked route from one street to another.

Statistics.py: Produces various plots to provide general information about crashes and traffic volumes.

Bonus.py: Executes a machine learning linear regression model on housing prices and provides statistical data about the results.

Datasets
collision_data_2017.csv: Contains data on traffic crashes that occurred in New York City during 2017.

dataset4.csv: Contains data on traffic volumes in New York City for the year 2017.

2017_NYC_Weather_Collision.csv: A combined dataset that includes crash data along with corresponding weather conditions at the time of the incidents.

Housing.csv: Contains data on housing prices in New York City.

Usage
Ensure that you have Python installed on your machine along with the necessary libraries (e.g., pandas, matplotlib, scikit-learn, etc.).
Download or clone this repository to your local machine.
Navigate to the directory where the scripts are located using your command line interface.
Execute the desired script by running python <script_name.py>.
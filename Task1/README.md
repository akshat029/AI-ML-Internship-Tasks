Task 1: Titanic Dataset - Data Cleaning and Preprocessing

This task involved cleaning and preprocessing the Titanic dataset to prepare it for further analysis or machine learning modeling. The steps were performed using Python and libraries like Pandas, Seaborn, and Matplotlib.

🔍 Dataset Information
The dataset contains information about passengers on the Titanic, such as:

Survival status

Passenger class

Name

Sex

Age

Number of siblings/spouses aboard

Number of parents/children aboard

Ticket info

Fare

Cabin

Embarked port

✅ Steps Performed
Loaded the dataset using pandas.read_csv().

Explored basic structure using .head() and .info() to understand data types and missing values.

Handled missing values:

Replaced missing Age values with the median.

Replaced missing Embarked values with the most frequent value (mode).

Dropped columns with excessive missing values: Cabin, Ticket.

Dropped unnecessary columns like PassengerId.

Normalized numerical columns (Age and Fare) using StandardScaler.

Visualized data distribution using boxplots to detect outliers.

Removed outliers based on IQR (Interquartile Range) for Age and Fare.

📊 Libraries Used
pandas

numpy

seaborn

matplotlib

sklearn.preprocessing.StandardScaler

📁 Output
A cleaned DataFrame, ready for analysis or machine learning tasks.

Boxplots showing distribution before outlier removal.

📎 How to Run:
Make sure you have the following Python packages installed:
pip install pandas numpy matplotlib seaborn scikit-learn

Then run:
python titanic_cleanup.py

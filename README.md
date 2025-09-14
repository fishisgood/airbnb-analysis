# airbnb-analysis

This project analyzes Airbnb listings data for a selected city, exploring relationships between variables such as price, neighbourhood, hosts, and more. The analysis includes data cleaning, feature engineering, and visualizations.

## Getting Started

1. **Download Data**  
   Download the `listings.csv` file for your city from the official Airbnb website.

2. **Project Structure**  
   Place the `listings.csv` file in the `data` folder inside the project directory:
   ```
   airbnb-analysis/
   ├── data/
   │   └── listings.csv
   ├── outputs/
   ├── analysis.py
   └── README.md
   ```

3. **Install Dependencies**  
   Install required Python packages:
   ```
   pip install -r requirements.txt
   ```

4. **Run the Analysis**  
   Run the analysis script in your local environment (VS Code, Jupyter, etc.):
   ```
   python analysis.py
   ```

## Output

- Cleaned data will be saved to the `outputs` folder.
- Visualizations and summary statistics will be displayed during execution.

## Notes

- No Google Colab or Drive code is used; all paths are local and relative.
- Make sure to exclude large data files and outputs from version control using `.gitignore`.

## Requirements

See `requirements.txt` for the full list of dependencies.
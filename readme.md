# Dicoding Ecommerce Analysis Dashboard

## Folder Structure

```plaintext
📂 ECommerce_olist_analysis
├── 📂 dashboard               # Streamlit dashboard scripts
├─────📄 dashboard.py          # Main dashboard script for Streamlit
├── 📂 data                    # Directory containing CSV files
├── 📄 main.ipynb          # Jupyter notebooks for data analysis and visualization
├── 📄 README.md               # Overview of the project (this file)
├── 📄 requirements.txt        # List of Python packages required for the project
```

### Prerequisites

Make sure you have **Python 3.12** installed along with the following libraries:
-matplotlib
-pandas
-seaborn
-streamlit
-numpy

You can install all dependencies using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Running the Dashboard

1. Clone the repository:

```bash
git clone https://github.com/MFavianZaahir/tugas_akhir
cd air-quality-analysis
```

2. Run the Streamlit dashboard:

```bash
cd dashboard
streamlit run dashboard/dashboard.py
```

Checkout the website: https://airqualityanalytics.streamlit.app/
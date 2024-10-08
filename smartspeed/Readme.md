## Smartspeed Integration

### Description
This repository demonstrates the integration of **SmartSpeed** with the `vald` class, used for managing test data retrieval, and processing, from Vald's external Smartspeed, Tenants, and Profile APIs. 

### Features
- Fetch test metrics from the **SmartSpeed API**
- Retrieve corresponding athlete and group (team) information from the **Profiles API** and **Tenants API**
- Organize and export data to CSV files:
    - Create different folders for different sports
    - Create separate CSV files for each test type

### Installation

1. **Clone the repository:**
    ```bash
    git clone <repository-url>
    ```

2. **Navigate to the project directory:**
    ```bash
    cd Smartspeed-Integration
    ```

3. **Install required dependencies:**
    Ensure you have Python installed along with the necessary libraries. You can install the required packages by running:
    ```bash
    pip install -r requirements.txt
    ```

4. **Create a `.env` file:**
    In the project directory, create a `.env` file to securely store your API credentials:
    ```bash
    touch .env
    ```

    Add the following content to the `.env` file:
    ```
    CLIENT_ID=<your-client-id>
    CLIENT_SECRET=<your-client-secret>
    TENANT_ID=<your-tenant-id>
    ```

    Replace `<your-client-id>`, `<your-client-secret>`, and `<your-tenant-id>` with your actual login information.

### Usage

1. **Option 1: Run the Jupyter Notebook for a Detailed Walkthrough**
    - Launch the notebook using Jupyter:
    ```bash
    jupyter notebook Smartspeed Integration.ipynb
    ```
    - The notebook guides you through connecting to the **SmartSpeed API**, retrieving athlete data, and performing test processing and analysis.
    - It helps understand how the code works and allows for customization.

2. **Option 2: Automatically Populate Folders Using `main_smartspeed.py`**
    - Alternatively, if you don't want to learn the details of the code and just need the output, run the `main_smartspeed.py` script.
    - This script will automatically fetch the data, process it, and populate the folders for different sports and test types (assuming you have created the `.env` file with your credentials).
    - Run the script from the command line:
    ```bash
    python main_smartspeed.py
    ```

### File Structure
- `Smartspeed Integration.ipynb`: Jupyter notebook containing the code for SmartSpeed data integration.
- `requirements.txt`: List of required Python packages for running the notebook.
- `vald_smartspeed.py`: Python file containing the Vald class and it's attributes and methods. 
- `main_smartspeed.py`: Python file that will auto-populate your Smartspeed data in the directory where it is run.
- `.env`: Environment file for storing API credentials (not included, you must create your own).



### Contributors
- [Nicholas Bettencourt](https://github.com/nbetts2020)

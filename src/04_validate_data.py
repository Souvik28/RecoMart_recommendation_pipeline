import pandas as pd
import pandera.pandas as pa
from pandera import Column, Check, DataFrameSchema
import logging
import os
import glob
from fpdf import FPDF, XPos, YPos
from datetime import datetime

# --- Configuration ---
LAKE_PATH = "../recomart_lake/raw/internal/transactions/*/*.csv"
REPORT_OUTPUT = "../recomart_lake/reports/data_quality_report.pdf"
os.makedirs("../recomart_lake/reports", exist_ok=True)

# Define the schema contract for RecoMart Interactions [cite: 248, 250]
interaction_schema = DataFrameSchema({
    "user_id": Column(str, nullable=False),
    "product_id": Column(str, nullable=False),
    "rating": Column(float, Check.in_range(1, 5), nullable=True),
    "timestamp": Column(pa.DateTime, nullable=False)
})

class DQReport(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 15)
        self.cell(0, 10, 'RecoMart Data Quality Report', new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        self.set_font('helvetica', '', 10)
        self.cell(0, 10, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='R')
        self.ln(5)

    def add_section(self, title, content):
        self.set_font('helvetica', 'B', 12)
        self.cell(0, 10, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
        self.set_font('helvetica', '', 10)
        self.multi_cell(0, 8, content)
        self.ln(5)

def generate_pdf_report(metrics, failure_cases, file_name):
    pdf = DQReport()
    pdf.add_page()
    
    # Section 1: Summary Metrics 
    summary = (f"Source File: {file_name}\n"
               f"Total Rows Processed: {metrics['total_rows']}\n"
               f"Duplicate Entries Found: {metrics['duplicates']}\n"
               f"Missing Values: {metrics['missing_values']}\n"
               f"Validation Status: {'FAILED' if failure_cases is not None else 'PASSED'}")
    pdf.add_section("1. Key Quality Metrics", summary)

    # Section 2: Failure Details [cite: 252]
    if failure_cases is not None:
        cases_text = failure_cases.to_string()
        pdf.add_section("2. Schema & Range Violations", f"The following rows failed validation:\n\n{cases_text}")
    else:
        pdf.add_section("2. Schema & Range Violations", "No violations found. Data conforms to schema.")

    pdf.output(REPORT_OUTPUT)
    print(f"PDF Report generated: {REPORT_OUTPUT}")

def validate_and_profile(file_path):
    df = pd.read_csv(file_path, parse_dates=['timestamp'])
    
    # Calculate Metrics [cite: 249]
    metrics = {
        'total_rows': len(df),
        'duplicates': df.duplicated().sum(),
        'missing_values': df.isnull().sum().sum()
    }
    
    failure_cases = None
    try:
        interaction_schema.validate(df, lazy=True)
        print(f"Validation Passed for {file_path}")
    except pa.errors.SchemaErrors as err:
        print(f"Validation Failed")
        failure_cases = err.failure_cases

    # Trigger PDF Generation 
    generate_pdf_report(metrics, failure_cases, os.path.basename(file_path))

if __name__ == "__main__":
    # Check for combined data first, then fall back to batch data
    combined_path = "../recomart_lake/processed/combined_transactions.csv"
    
    if os.path.exists(combined_path):
        print("Validating combined streaming + batch data")
        validate_and_profile(combined_path)
    else:
        print("Validating batch data only")
        csv_files = glob.glob(LAKE_PATH)
        if csv_files:
            latest_file = max(csv_files, key=os.path.getctime)
            validate_and_profile(latest_file)
        else:
            print("No files found to validate.")
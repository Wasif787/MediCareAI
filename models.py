#models.py
import re
import joblib
import pandas as pd
from PyPDF2 import PdfReader
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def extract_data(text):
    """Common data extraction function for all models"""
    # Use regular expressions to extract all features
    mr_match = re.search(r"MR #\s*[:\-]?\s*(\d+)", text)
    age_match = re.search(r"Age\s*:\s*(\d+\s*Year,\d+\s*Month,\d+\s*Days)", text)
    gender_match = re.search(r"Gender\s*:\s*(M|F)", text)
    haemoglobin_match = re.search(r"Haemoglobin\s*(\d+\.\d+)", text)
    redCellCount_match = re.search(r"Red Cell Count\s*(\d+\.\d+)", text)
    pcv_match = re.search(r"P.C.V.\s*(\d+\.\d+)", text)
    mcv_match = re.search(r"M.C.V.\s*(\d+\.\d+)", text)
    mch_match = re.search(r"M.C.H.\s*(\d+\.\d+)", text)
    mchc_match = re.search(r"M.C.H.C.\s*(\d+\.\d+)", text)
    rdwCV_match = re.search(r"RDW-CV\s*(\d+\.\d+)", text)
    totalWBCcount_match = re.search(r"Total WBC Count\s*(\d+\.\d+)", text)
    neutrophil_match = re.search(r"Neutrophil\s*(\d+\.\d+)", text)
    lymphocytes_match = re.search(r"Lymphocytes\s*(\d+\.\d+)", text)
    monocyte_match = re.search(r"Monocyte\s*(\d+\.\d+)", text)
    eosinophil_match = re.search(r"Eosinophil\s*(\d+\.\d+)", text)
    basophil_match = re.search(r"Basophil\s*(\d+\.\d+)", text)
    platelet_match = re.search(r"Platelet Count\s*(\d+)", text)
    rbc_morphology_match = re.search(r"RBC Morphology\s*:\s*([\w, ]+)", text)

    # Extract values
    extracted_values = {
        'mr_number': mr_match.group(1) if mr_match else "Not Found",
        'age': age_match.group(1) if age_match else "Not Found",
        'gender': gender_match.group(1) if gender_match else "Not Found",
        'haemoglobin': haemoglobin_match.group(1) if haemoglobin_match else "Not Found",
        'redCellCount': redCellCount_match.group(1) if redCellCount_match else "Not Found",
        'pcv': pcv_match.group(1) if pcv_match else "Not Found",
        'mcv': mcv_match.group(1) if mcv_match else "Not Found",
        'mch': mch_match.group(1) if mch_match else "Not Found",
        'mchc': mchc_match.group(1) if mchc_match else "Not Found",
        'rdwCV': rdwCV_match.group(1) if rdwCV_match else "Not Found",
        'totalWBCcount': totalWBCcount_match.group(1) if totalWBCcount_match else "Not Found",
        'neutrophil': neutrophil_match.group(1) if neutrophil_match else "Not Found",
        'lymphocytes': lymphocytes_match.group(1) if lymphocytes_match else "Not Found",
        'monocyte': monocyte_match.group(1) if monocyte_match else "Not Found",
        'eosinophil': eosinophil_match.group(1) if eosinophil_match else "Not Found",
        'basophil': basophil_match.group(1) if basophil_match else "Not Found",
        'platelet': platelet_match.group(1) if platelet_match else "Not Found",
        'rbc_morphology': rbc_morphology_match.group(1).strip() if rbc_morphology_match else "Not Found"
    }

    return extracted_values

def preprocess_hemophilia_data(data):
    """Preprocess data for hemophilia prediction"""
    df = pd.DataFrame({
        'Haemoglobin': [data['haemoglobin']],
        'redCellCount': [data['redCellCount']],
        'pcv': [data['pcv']],
        'mch': [data['mch']],
        'mchc': [data['mchc']],
        'rdwCV': [data['rdwCV']],
        'totalWBCcount': [data['totalWBCcount']],
        'neutrophil': [data['neutrophil']],
        'lymphocytes': [data['lymphocytes']],
        'monocyte': [data['monocyte']]
    })
    return process_dataframe(df)

def preprocess_anemia_data(data):
    """Preprocess data for anemia prediction"""
    df = pd.DataFrame({
        'Haemoglobin': [data['haemoglobin']],
        'redCellCount': [data['redCellCount']],
        'mcv': [data['mcv']],
        'mch': [data['mch']],
        'mchc': [data['mchc']],
        'platelet': [data['platelet']]
    })
    return process_dataframe(df)

def preprocess_thalassemia_data(data):
    """Preprocess data for thalassemia prediction"""
    df = pd.DataFrame({
        'Gender': [data['gender']],
        'C_Hemoglobin': [data['haemoglobin']],
        'C_Red Cell Count': [data['redCellCount']],
        'C_PCV': [data['pcv']],
        'C_MCV': [data['mcv']],
        'C_MCH': [data['mch']],
        'C_MCHC': [data['mchc']],
        'C_RDW-CV': [data['rdwCV']]
    })
    df['Gender'] = df['Gender'].map({'M': 0, 'F': 1})
    return process_dataframe(df)

def preprocess_fibrinogen_data(data):
    """Preprocess data for fibrinogen prediction"""
    df = pd.DataFrame({
        'Haemoglobin': [data['haemoglobin']],
        'Red Cell Count': [data['redCellCount']],
        'P.C.V.': [data['pcv']],
        'M.C.V.': [data['mcv']],
        'M.C.H.': [data['mch']],
        'M.C.H.C.': [data['mchc']],
        'RDW-CV': [data['rdwCV']],
        'Total WBC Count': [data['totalWBCcount']],
        'Neutrophil': [data['neutrophil']],
        'Lymphocytes': [data['lymphocytes']],
        'Platelet Count': [data['platelet']]
    })
    return process_dataframe(df)

def process_dataframe(df):
    """Common DataFrame processing steps"""
    df.replace('Not Found', np.nan, inplace=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    return df

def predict_disease(pdf_path, model_choice):
    """Main prediction function"""
    try:
        # Read PDF
        reader = PdfReader(pdf_path)
        text = "".join(page.extract_text() for page in reader.pages)
        
        # Extract features
        extracted_data = extract_data(text)
        
        # Select model and preprocess data based on choice
        if model_choice == 1:
            print("\nClass 0: Low APTT (thrombosis) | Class 1: Normal APTT | Class 2: High APTT (Hemophilia)")
            model = joblib.load('models/hemophilia_naive_bayes_model.pkl')
            processed_data = preprocess_hemophilia_data(extracted_data)
            classes = ["Class 0", "Class 1", "Class 2"]
            
        elif model_choice == 2:
            print("\nClass 0: Normocytic,Normochromic | Class 1: Normocytic | Class 2: Microcytosis | Class 3: Hypochromia | Class 4: Macrocytosis | Class 5: Macrocytic | Class 6: Isopoikilocytosis")
            model = joblib.load('models/anemia_random_forest_model.pkl')
            processed_data = preprocess_anemia_data(extracted_data)
            classes = list(range(7))
            
        elif model_choice == 3:
            model = joblib.load('models/thalsemia_random_forest_model.pkl')
            processed_data = preprocess_thalassemia_data(extracted_data)
            classes = list(range(5))  # Now handles 5 classes
            
        elif model_choice == 4:
            model = joblib.load('models/fibrongen_knn_model.pkl')
            processed_data = preprocess_fibrinogen_data(extracted_data)
            classes = list(range(3))  # Now handles 3 classes
            
        else:
            raise ValueError("Invalid model choice")

        # Make prediction
        probabilities = model.predict_proba(processed_data)
        
        # Print results
        print("\nPrediction Results:")
        for i, prob in enumerate(probabilities[0]):
            print(f"Class {i} probability: {prob:.4f}")
        
        return probabilities

    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def main():
    while True:
        print("\nDisease Prediction System")
        print("=" * 30)
        print("1. Hemophilia Prediction")
        print("2. Anemia Prediction")
        print("3. Thalassemia Prediction")
        print("4. Fibrinogen Prediction")
        print("5. Exit")
        
        try:
            choice = int(input("\nEnter your choice (1-5): "))
            
            if choice == 5:
                print("Thank you for using the Disease Prediction System!")
                break
                
            if choice not in [1, 2, 3, 4]:
                print("Invalid choice. Please select a number between 1 and 5.")
                continue
                
            pdf_path = input("Enter the path to your CBC PDF file: ")
            predict_disease(pdf_path, choice)
            
        except ValueError:
            print("Invalid input. Please enter a number.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
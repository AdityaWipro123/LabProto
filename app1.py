import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random
from math import sqrt
from sklearn.metrics.pairwise import euclidean_distances


groupings = {
    "Basic Dimensions": ['STROKE', 'BORE', 'ROD DIA', 'Cushioning', 'Working Pressure'],
    "Rod": ['Rod_Piston_Engage_Dia(d_r)', 'Rod_Piston_Engage_Length(e_r)'],
    "Rod Eye": ['Rod_Eye_Thickness(OD-ID/2)(t_re)', 'Pin_Dia(dp_re)', 'Pin_width(m_re)', 'Rod_Eye_Least_Thickness(t_l)'],
    "Piston": ['Piston_Thickness(o-i/2)(t_p)', 'Piston_Length(l_p)'],
    "Tube": ['Thickness(o-i/2)(t_t)', 'Length(l_t)'],
    "CEC": ['CEC_Thickness(OD-ID)(t_c)', 'Pin_Dia(dp)', 'Pin_width(m_c)', 'Counterbore_ratio(r_c)'],
    "HEC": ['HEC_Inner_dia(i_h)', 'Engage_Length(e_h)'],
    "CEC-Tube weld": ['CEC-Tube_weld_Strength(Kg/sqmm)', 'CEC-Tube_weld_Radius(R)', 'CEC-Tube_weld_angle(a)', 'CEC-Tube_weld_depth(d)'],
    "HEC Port": ['Port_hole_dia_RHS(h_r)', 'Port_SpotFace_dia_RHS(d_r)', 'Port_spotface_width_RHS(w_r)'],
    "CEC port": ['Where(CEC_or_Tube)', 'port_hole_dia_near_CEC(h_l)', 'Port_Spot_Face_dia_near_CEC(d_l)'],
    "Test Parameters": ['Test_Pressure', 'Target']
}

# Streamlit page setup
st.set_page_config(page_title="app", layout="wide")


import os
from PIL import Image
import sys  # Make sure this is already imported at the top
base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))

logo_path = os.path.join(base_path, "WiproHydraulics.png")

# Use your specified path
# logo_path = r"C:\Users\hi80050775\OneDrive - Wipro\Desktop\Models\WiproHydraulics.png"  # ← replace with your actual path

from PIL import Image
import base64

# Load and convert image to base64
with open(logo_path, "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode()

# HTML block to center logo and header
st.markdown(f"""
    <div style='text-align: center;'>
        <img src="data:image/jpeg;base64,{encoded_image}" width="600"/>
        <h1 style='color: #003366;'>Product Validation using AI</h1>

    </div>
""", unsafe_allow_html=True)


# Load model and preprocessing bundles
nn_path = os.path.join(base_path, "nn_bundle_2025_07_15_12_19_04.pkl")
pp_path = os.path.join(base_path, "preprocessing_bundle_2025_07_15_14_53_37.pkl")
preprocess_bundle = joblib.load(pp_path)
nn_bundle = joblib.load(nn_path)



# Extract components from bundles
nn_model = nn_bundle["model"]
scaler = preprocess_bundle["scaler"]
ohe = preprocess_bundle["ohe"]
cat_cols = preprocess_bundle["cat_cols"]
num_cols = preprocess_bundle["num_cols"]
nn_features = preprocess_bundle["feature_names"]

# Load template and original historical 
knn_path = os.path.join(base_path, "knn_bundle_2025_07_15_14_55_43.pkl" )
knn_bundle = joblib.load(knn_path)
# knn_bundle = joblib.load(r"C:\Users\hi80050775\OneDrive - Wipro\Desktop\Models\knn_bundle_2025_07_11_17_15_01.pkl")
X = knn_bundle['X']
X_processed = knn_bundle['X_processed']
data = knn_bundle['data']
full_data = X.reset_index(drop=True)
k = knn_bundle['optimal_k']

# Initialize prefill row
if 'prefill_row' not in st.session_state:
    st.session_state.prefill_row = full_data.sample(n=1, random_state=random.randint(1, 10000)).iloc[0]
prefill_row = st.session_state.prefill_row

# Yield/Shear Strength Tables
strength_options = {
    'C45-Round': {'No_Q&T': {'Tensile': 3600, 'Shear': 3780}, 'Q&T': {'Tensile': 5000, 'Shear': 4200}},
    'C45-Forge': {'No_Q&T': {'Tensile': 3400, 'Shear': 3780}, 'Q&T': {'Tensile': 5000, 'Shear': 4200}},
    'C45 - Gas cutting': {'No_Q&T': {'Tensile': 3300, 'Shear': 3480}, 'Q&T': {'Tensile': 3600, 'Shear': 3600}},
    'C45 - CDS/DOM': {'No_Q&T': {'Tensile': 5000, 'Shear': 3600}, 'Q&T': {'Tensile': 5000, 'Shear': 3600}},
    'Alloy steel - RR': {'No_Q&T': {'Tensile': 5300, 'Shear': 4110}, 'Q&T': {'Tensile': 5300, 'Shear': 4110}}
}

# Layout: two columns
input_dict = {}
left_col, right_col = st.columns([1, 1], gap="large")




import os
from PIL import Image
import sys  # Make sure this is already imported at the top
# image_path = os.path.join(base_path, image_filename)
# Get the correct base path (for .exe compatibility too)
base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))

# Input form
with left_col:
    st.subheader("Enter Cylinder Parameters")
    for section, fields in groupings.items():
        with st.expander(section, expanded=True):
            for field in fields:
                key = f'input_{field}'
                if field in num_cols:
                    input_dict[field] = st.number_input(label=field, value=float(prefill_row[field]), key=key)
                elif field in cat_cols:
                    options = X[field].dropna().unique().tolist()
                    default_index = options.index(prefill_row[field]) if prefill_row[field] in options else 0
                    input_dict[field] = st.selectbox(label=field, options=options, index=default_index, key=key)
                else:
                    input_dict[field] = st.number_input(label=field,value=float(35),  key=key)
                    
        image_filename = f"{section}.jpg"
        image_path = os.path.join(base_path, image_filename)
        if os.path.exists(image_path):
            if st.checkbox(f"Show {section} Diagram", key=f"{section}_checkbox"):
                st.image(Image.open(image_path), caption=f"{section} Diagram", use_container_width=True)
    

# Prediction block
with right_col:
    st.subheader("Select Material Options")
    material_parts = ['Rod', 'Rod_Eye', 'Piston', 'Tube', 'CEC', 'HEC']
    material_config = {}
    for part in material_parts:
        with st.expander(f"Material for {part}"):
            mat_key = f"{part}_Material"
            qt_key = f"{part}_QT"
            default_mat = 'C45-Round'
            default_qt = 'No_Q&T'
            material_config[mat_key] = st.selectbox(f"{part} Material", options=list(strength_options.keys()), index=0, key=mat_key)
            material_config[qt_key] = st.selectbox(f"{part} Condition", options=['No_Q&T', 'Q&T'], index=0, key=qt_key)
    
    st.subheader("Factor of Safety (FOS) Summary")
    # Extract input features used in FOS calculations
    BORE = input_dict['BORE']
    ROD_DIA = input_dict['ROD DIA']
    STROKE = input_dict['STROKE']
    Working_Pressure = input_dict['Working Pressure']
    Test_Pressure = input_dict['Test_Pressure']
    Thickness = input_dict['Thickness(o-i/2)(t_t)']
    Pin_width_RE = input_dict['Pin_width(m_re)']
    Rod_Piston_Engage_Dia = input_dict['Rod_Piston_Engage_Dia(d_r)']
    Rod_Piston_Engage_Length = input_dict['Rod_Piston_Engage_Length(e_r)']
    Rod_Eye_Thickness = input_dict['Rod_Eye_Thickness(OD-ID/2)(t_re)']
    Pin_dia_RE = input_dict['Pin_Dia(dp_re)']
    Rod_Eye_Least_Thickness = input_dict['Rod_Eye_Least_Thickness(t_l)']
    Piston_Length = input_dict['Piston_Length(l_p)']
    Piston_Thickness = input_dict['Piston_Thickness(o-i/2)(t_p)']
    Pin_dia_CEC = input_dict['Pin_Dia(dp)']
    CEC_Thickness = input_dict['CEC_Thickness(OD-ID)(t_c)']
    Pin_width_CEC = input_dict['Pin_width(m_c)']
    HEC_Inner_dia = input_dict['HEC_Inner_dia(i_h)']
    Engage_Length = input_dict['Engage_Length(e_h)']
    CEC_Tube_weld_Strength = input_dict['CEC-Tube_weld_Strength(Kg/sqmm)']
    Port_hole_dia_RHS = input_dict['Port_hole_dia_RHS(h_r)']
    port_hole_dia_near_CEC = input_dict['port_hole_dia_near_CEC(h_l)']

    # Assign strengths based on material config
    Rod_Tensile = strength_options[material_config['Rod_Material']][material_config['Rod_QT']]['Tensile']
    Rod_Shear = strength_options[material_config['Rod_Material']][material_config['Rod_QT']]['Shear']
    Eye_Tensile = strength_options[material_config['Rod_Eye_Material']][material_config['Rod_Eye_QT']]['Tensile']
    Eye_Shear = strength_options[material_config['Rod_Eye_Material']][material_config['Rod_Eye_QT']]['Shear']
    Piston_Tensile = strength_options[material_config['Piston_Material']][material_config['Piston_QT']]['Tensile']
    Piston_Shear = strength_options[material_config['Piston_Material']][material_config['Piston_QT']]['Shear']
    Tube_Tensile = strength_options[material_config['Tube_Material']][material_config['Tube_QT']]['Tensile']
    CEC_Tensile = strength_options[material_config['CEC_Material']][material_config['CEC_QT']]['Tensile']
    CEC_Shear = strength_options[material_config['CEC_Material']][material_config['CEC_QT']]['Shear']
    HEC_Tensile = strength_options[material_config['HEC_Material']][material_config['HEC_QT']]['Tensile']
    HEC_Shear = strength_options[material_config['HEC_Material']][material_config['HEC_QT']]['Shear']

    # Proceed with FOS calculations using the above values
    # ... (your FOS calculation logic here) ...

    
    # Constants
    E = 2100000
    F = (np.pi/4) * (BORE)**2 * Working_Pressure
    Fex = (np.pi/4) * ((BORE)**2 - (ROD_DIA)**2) * Working_Pressure

    # Rod
    
    Rod_Axial_FOS = (Rod_Tensile * np.pi * (ROD_DIA)**2) / (4 * Fex)
    Rod_Shear_FOS = (Rod_Shear * np.pi * Rod_Piston_Engage_Dia * Rod_Piston_Engage_Length) / (2 * F)
    Rod_Buckling_FOS = (np.pi**3 * E * ROD_DIA**4) / (64 * F * STROKE**2)

    # Rod Eye
    
    Eye_Thickness_FOS = Eye_Tensile / (F / (2 * Rod_Eye_Thickness * Pin_width_RE))
    Least_Eye_FOS = Eye_Tensile / (F / (2 * Rod_Eye_Least_Thickness * Pin_width_RE))
    Eye_Shear_FOS = Eye_Shear / (Fex / (2 * Pin_width_RE * sqrt(((Pin_dia_RE/2) + Rod_Eye_Thickness)**2 - (Pin_dia_RE/2)**2)))

    # Piston
   
    Piston_Axial_FOS = Piston_Tensile / (4 * F / (np.pi * (BORE**2 - Rod_Piston_Engage_Dia**2)))
    Piston_Shear_FOS = Piston_Shear / (F / (Piston_Length * np.pi * Rod_Piston_Engage_Dia))

    # Tube
    
    Tube_Axial_FOS = Tube_Tensile / (4 * F / (np.pi * ((BORE + 2*Thickness)**2 - BORE**2)))
    Tube_Circum_FOS = Tube_Tensile / (Working_Pressure * (BORE + 2*Thickness) / (2*Thickness))

    # CEC
    
    CEC_Tensile_FOS = CEC_Tensile / (Fex / (2 * CEC_Thickness * Pin_width_CEC))
    CEC_Shear_FOS = CEC_Shear / (Fex / (2 * Pin_width_CEC * sqrt(((Pin_dia_CEC/2) + CEC_Thickness)**2 - (Pin_dia_CEC/2)**2)))

    # HEC
    
    HEC_Tensile_FOS = (HEC_Tensile * np.pi * ((BORE)**2 - (HEC_Inner_dia)**2)) / (4 * F)
    HEC_Shear_FOS = (HEC_Shear * np.pi * Engage_Length * BORE) / F

    # Weld
    Tube_CECWeld_FOS = (CEC_Tube_weld_Strength * 100 * np.pi * (((BORE + 2 * Thickness)**2 - (BORE + 1)**2))) / (4 * F)

    # Port
    Port_Circum_HEC = Tube_Tensile / (Working_Pressure * (BORE + 2*Thickness) / (2*Thickness)) * (1 + Port_hole_dia_RHS / (2*Thickness))
    Port_Circum_CEC = Tube_Tensile / (Working_Pressure * (BORE + 2*Thickness) / (2*Thickness)) * (1 + port_hole_dia_near_CEC / (2*Thickness))


    with st.container():       
        # st.markdown("### Factor of Safety (FOS) Summary")
        # Toggle this variable to control interactivity
        is_editable = True  # Set to False if you want to disable input fields
        
        # ROD FOS
        st.subheader("Rod FOS")
        col1, col2 = st.columns(2)
        with col1:
            st.text_input("Rod_Axial_FOS:", value=f"{Rod_Axial_FOS:.1f}", disabled=not is_editable)
            st.text_input("Rod_Shear_FOS:", value=f"{Rod_Shear_FOS:.1f}", disabled=not is_editable)
            st.text_input("Rod_Buckling_FOS:", value=f"{Rod_Buckling_FOS:.1f}", disabled=not is_editable)
        with col2:
            st.text_input("Desired_Rod_Tensile_FOS:", value="4.00", disabled=not is_editable)
            st.text_input("Desired_Rod_Shear_FOS:", value="7.00", disabled=not is_editable)
            st.text_input("Desired_Rod_Buckling_FOS:", value="4.00", disabled=not is_editable)
        
        # ROD EYE FOS
        st.subheader("Rod Eye FOS")
        col1, col2 = st.columns(2)
        with col1:
            st.text_input("Eye_Thickness_FOS:", value=f"{Eye_Thickness_FOS:.2f}", disabled=not is_editable)
            st.text_input("Leasteye_Thickness_FOS:", value=f"{Least_Eye_FOS:.2f}", disabled=not is_editable)
            st.text_input("Eye_Shear_FOS:", value=f"{Eye_Shear_FOS:.2f}", disabled=not is_editable)
        with col2:
            st.text_input("Desired_Eye_FOS:", value="4.00", disabled=not is_editable)
            st.text_input("Desired_Least_Eye_FOS:", value="4.00", disabled=not is_editable)
            st.text_input("Desired_Eye_Shear_FOS:", value="7.00", disabled=not is_editable)

        
        # PISTON FOS
        st.subheader("Piston FOS")
        col1, col2 = st.columns(2)
        with col1:
            st.text_input("Piston_Axial_FOS:", value=f"{Piston_Axial_FOS:.2f}", disabled=not is_editable)
            st.text_input("Piston_Shear_FOS:", value=f"{Piston_Shear_FOS:.2f}", disabled=not is_editable)
        with col2:
            st.text_input("Desired_Piston_Tensile_FOS:", value="4.00", disabled=not is_editable)
            st.text_input("Piston_Piston_Shear_FOS:", value="7.00", disabled=not is_editable)
        
        # TUBE FOS
        st.subheader("Tube FOS")
        col1, col2 = st.columns(2)
        with col1:
            st.text_input("Tube_Axial_FOS:", value=f"{Tube_Axial_FOS:.2f}", disabled=not is_editable)
            st.text_input("Tube_Circum_FOS:", value=f"{Tube_Circum_FOS:.2f}", disabled=not is_editable)
        with col2:
            st.text_input("Desired_Axial_FOS:", value="4.00", disabled=not is_editable)
            st.text_input("Desired_Circum_FOS:", value="2.00", disabled=not is_editable)
        
        # CEC FOS
        st.subheader("CEC FOS")
        col1, col2 = st.columns(2)
        with col1:
            st.text_input("CEC_Tensile_FOS:", value=f"{CEC_Tensile_FOS:.2f}", disabled=not is_editable)
            st.text_input("CEC_Shear_FOS:", value=f"{CEC_Shear_FOS:.2f}", disabled=not is_editable)

        with col2:
            st.text_input("Desired_CEC_Tensile_FOS:", value="4.00", disabled=not is_editable)
            st.text_input("Desired_CEC_Shear_FOS:", value="7.00", disabled=not is_editable)
        
        # HEC FOS
        st.subheader("HEC FOS")
        col1, col2 = st.columns(2)
        with col1:
            st.text_input("HEC_Tensile_FOS:", value=f"{HEC_Tensile_FOS:.2f}", disabled=not is_editable)
            st.text_input("HEC_Shear_FOS:", value=f"{HEC_Shear_FOS:.2f}", disabled=not is_editable)
        with col2:
            st.text_input("Desired_HEC_Tensile_FOS:", value="4.00", disabled=not is_editable)
            st.text_input("Desired_HEC_Shear_FOS:", value="7.00", disabled=not is_editable)
        
        # TUBE-CEC WELD FOS
        st.subheader("Tube-CEC Weld FOS")
        col1, col2 = st.columns(2)
        with col1:
            st.text_input("Tube_CECWeld_FOS:", value=f"{Tube_CECWeld_FOS:.2f}", disabled=not is_editable)
        with col2:
            st.text_input("Desired_Tube_CEC_Weld_FOS:", value="7.00", disabled=not is_editable)
        
        # PORT FOS
        st.subheader("Port FOS")
        col1, col2 = st.columns(2)
        with col1:
            st.text_input("HECsideHole_Circum_FOS:", value=f"{Port_Circum_HEC:.2f}", disabled=not is_editable)
            st.text_input("CECsideHole_Circum_FOS:", value=f"{Port_Circum_CEC:.2f}", disabled=not is_editable)
        with col2:
            st.text_input("Desired_HECsideHole_FOS:", value="2.00", disabled=not is_editable)
            st.text_input("Desired_CECsideHole_FOS:", value="2.00", disabled=not is_editable)





    
    # import io

    # # --- Prepare Cylinder Details Sheet ---
    # cylinder_df = pd.DataFrame(list(input_dict.items()), columns=["Parameter", "Value"])
    
    # # --- Prepare FOS Summary Sheet ---
    # fos_summary = {
    #     "Parameter": [
    #         "Rod_Axial_FOS", "Rod_Shear_FOS", "Rod_Buckling_FOS",
    #         "Eye_Thickness_FOS", "Leasteye_Thickness_FOS", "Eye_Shear_FOS",
    #         "Piston_Axial_FOS", "Piston_Shear_FOS",
    #         "Tube_Axial_FOS", "Tube_Circum_FOS",
    #         "CEC_Tensile_FOS", "CEC_Shear_FOS",
    #         "HEC_Tensile_FOS", "HEC_Shear_FOS",
    #         "Tube_CECWeld_FOS", "HECsideHole_Circum_FOS", "CECsideHole_Circum_FOS"
    #     ],
    #     "Calculated Value": [
    #         Rod_Axial_FOS, Rod_Shear_FOS, Rod_Buckling_FOS,
    #         Eye_Thickness_FOS, Least_Eye_FOS, Eye_Shear_FOS,
    #         Piston_Axial_FOS, Piston_Shear_FOS,
    #         Tube_Axial_FOS, Tube_Circum_FOS,
    #         CEC_Tensile_FOS, CEC_Shear_FOS,
    #         HEC_Tensile_FOS, HEC_Shear_FOS,
    #         Tube_CECWeld_FOS, Port_Circum_HEC, Port_Circum_CEC
    #     ],
    #     "Desired Value": [
    #         4, 7, 4, 4, 4, 7, 4, 7, 4, 2, 4, 7, 4, 7, 7, 2, 2
    #     ]
    # }
    # fos_df = pd.DataFrame(fos_summary)
    # # Define the columns to show for neighbor summary
    # selected_cols = ['Component', 'Cushioning', 'Working Pressure', 'STROKE', 'BORE', 'ROD DIA',
    #              'Test_Pressure', 'Target', 'Results', 'Part_Failed', 'Completed_Cycles', 'Report_No']

    # # --- Prepare Neighbor Data ---
    # neighbor_df_final = neighbor_data[selected_cols] if all(col in neighbor_data.columns for col in selected_cols) else neighbor_data
    
    # # --- Create Excel file with all 3 sheets ---
    # excel_buffer = io.BytesIO()
    # with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
    #     workbook = writer.book
    #     header_fmt = workbook.add_format({'bold': True, 'bg_color': '#D9E1F2'})
    
    #     # 1. Cylinder Details
    #     cylinder_df.to_excel(writer, sheet_name="Cylinder Details", index=False)
    #     ws1 = writer.sheets["Cylinder Details"]
    #     ws1.write(0, 0, "Parameter", header_fmt)
    #     ws1.write(0, 1, "Value", header_fmt)
    
    #     # 2. FOS Summary
    #     fos_df.to_excel(writer, sheet_name="FOS Summary", index=False)
    #     ws2 = writer.sheets["FOS Summary"]
    #     for col_num, value in enumerate(fos_df.columns.values):
    #         ws2.write(0, col_num, value, header_fmt)
    
    #     # 3. Passed-Failed Neighbors
    #     neighbor_df_final.to_excel(writer, sheet_name="Passed-Failed Neighbors", index=False)
    #     ws3 = writer.sheets["Passed-Failed Neighbors"]
    #     for col_num, value in enumerate(neighbor_df_final.columns.values):
    #         ws3.write(0, col_num, value, header_fmt)
    
    # excel_buffer.seek(0)
    
    # # --- Download button for final report ---
    # st.download_button(
    #     label="📄 Download Full FOS Report (Excel)",
    #     data=excel_buffer,
    #     file_name="FOS_Report_Full.xlsx",
    #     mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    # )

    
    
    
    
    # import pandas as pd
    # import io
    # import streamlit as st
    
    # # 1. Dummy FOS values (replace with real calculated values)
    # fos_summary = {
    #     "Parameter": [
    #         "Rod_Axial_FOS", "Rod_Shear_FOS", "Rod_Buckling_FOS",
    #         "Eye_Thickness_FOS", "Leasteye_Thickness_FOS", "Eye_Shear_FOS",
    #         "Piston_Axial_FOS", "Piston_Shear_FOS",
    #         "Tube_Axial_FOS", "Tube_Circum_FOS",
    #         "CEC_Tensile_FOS", "CEC_Shear_FOS",
    #         "HEC_Tensile_FOS", "HEC_Shear_FOS",
    #         "Tube_CECWeld_FOS", "HECsideHole_Circum_FOS", "CECsideHole_Circum_FOS"
    #     ],
    #     "Calculated Value": [
    #         5.1, 8.0, 4.2, 4.3, 4.0, 7.5, 4.7, 8.1, 4.2, 2.5,
    #         4.3, 7.2, 4.6, 7.8, 7.1, 2.2, 2.3
    #     ],
    #     "Desired Value": [
    #         4, 7, 4, 4, 4, 7, 4, 7, 4, 2, 4, 7, 4, 7, 7, 2, 2
    #     ]
    # }
    # fos_df = pd.DataFrame(fos_summary)
    
    # # 2. Convert input_dict to DataFrame
    # cylinder_df = pd.DataFrame(list(input_dict.items()), columns=["Parameter", "Value"])
    
    # # 3. Create Excel file in memory
    # excel_buffer = io.BytesIO()
    # with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
    #     workbook = writer.book
    
    #     # Write FOS Summary
    #     fos_df.to_excel(writer, sheet_name="FOS Summary", index=False, startrow=0)
    #     worksheet = writer.sheets["FOS Summary"]
    #     header_fmt = workbook.add_format({'bold': True, 'bg_color': '#D9E1F2'})
    #     for col_num, value in enumerate(fos_df.columns.values):
    #         worksheet.write(0, col_num, value, header_fmt)
    
    #     # Leave space and write Cylinder Details
    #     start_row = len(fos_df) + 3
    #     worksheet.write(start_row, 0, "Cylinder Details", header_fmt)
    #     cylinder_df.to_excel(writer, sheet_name="FOS Summary", startrow=start_row + 1, index=False)
    
    # excel_buffer.seek(0)
    
    # # 4. Download button in Streamlit
    # st.download_button(
    #     label="Download FOS Report with Cylinder Details",
    #     data=excel_buffer,
    #     file_name="FOS_Report.xlsx",
    #     mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    # )
  

# if st.button("Predict Result"):
#     st.subheader("Prediction")
#     input_df = pd.DataFrame([input_dict])
#     input_cat = ohe.transform(input_df[cat_cols])
#     cat_feature_names = ohe.get_feature_names_out(cat_cols)
#     X_cat_df = pd.DataFrame(input_cat, columns=cat_feature_names, index=input_df.index)
#     X_cat_df = X_cat_df.drop(columns=[cat_feature_names[0], cat_feature_names[-1]])

#     input_num = scaler.transform(input_df[num_cols])
#     X_num_df = pd.DataFrame(input_num, columns=num_cols, index=input_df.index)
#     input_processed = pd.concat([X_num_df, X_cat_df], axis=1)
#     for col in nn_features:
#         if col not in input_processed.columns:
#             input_processed[col] = 0
#     input_processed = input_processed[nn_features]

#     nn_prediction = nn_model.predict(input_processed)[0]
#     pred_label = "Failed" if nn_prediction == 1 else "Passed"
#     result_color = "red" if pred_label == "Failed" else "green"

#     st.markdown(f"""
#     <div style="padding: 30px; border-radius: 10px; background-color: {result_color}; color: white; text-align: center;">
#         <h2>{pred_label}</h2>
#     </div>
#     """, unsafe_allow_html=True)


#     # -------- Nearest Neighbors (NN) Using Euclidean Distance --------
#     st.subheader("Nearest Neighbors from Historical Data")
#     input_vector = input_processed.values
#     distances = euclidean_distances(X_processed[nn_features], input_vector).flatten()
#     top_k_indices = distances.argsort()[:k]
    
#     neighbor_df = data.iloc[top_k_indices]
#     selected_cols = ['Component', 'Cushioning', 'Working Pressure', 'STROKE', 'BORE', 'ROD DIA', 'Test_Pressure', 'Target', 'Results', 'Part_Failed', 'Completed_Cycles', 'Report_No']
#     neighbor_data = neighbor_df[selected_cols] if all(c in neighbor_df.columns for c in selected_cols) else neighbor_df
    
#     tab1, tab2 = st.tabs(["✅ Passed", "❌ Failed"])
#     with tab1:
#         passed = neighbor_data[neighbor_data['Results'] == 'Passed']
#         st.write(f"**{len(passed)} Passed Neighbors:**")
#         st.dataframe(passed)
    
#     with tab2:
#         failed = neighbor_data[neighbor_data['Results'] == 'Failed']
#         st.write(f"**{len(failed)} Failed Neighbors:**")
#         st.dataframe(failed)




# import io

# if st.button("📤 Generate Excel Report"):
#     # Convert user inputs to DataFrame
#     cylinder_df = pd.DataFrame(list(input_dict.items()), columns=["Parameter", "Value"])
    
#     # Recreate FOS summary from current calculated values
#     fos_summary = {
#         "Parameter": [
#             "Rod_Axial_FOS", "Rod_Shear_FOS", "Rod_Buckling_FOS",
#             "Eye_Thickness_FOS", "Leasteye_Thickness_FOS", "Eye_Shear_FOS",
#             "Piston_Axial_FOS", "Piston_Shear_FOS",
#             "Tube_Axial_FOS", "Tube_Circum_FOS",
#             "CEC_Tensile_FOS", "CEC_Shear_FOS",
#             "HEC_Tensile_FOS", "HEC_Shear_FOS",
#             "Tube_CECWeld_FOS", "HECsideHole_Circum_FOS", "CECsideHole_Circum_FOS"
#         ],
#         "Calculated Value": [
#             Rod_Axial_FOS, Rod_Shear_FOS, Rod_Buckling_FOS,
#             Eye_Thickness_FOS, Least_Eye_FOS, Eye_Shear_FOS,
#             Piston_Axial_FOS, Piston_Shear_FOS,
#             Tube_Axial_FOS, Tube_Circum_FOS,
#             CEC_Tensile_FOS, CEC_Shear_FOS,
#             HEC_Tensile_FOS, HEC_Shear_FOS,
#             Tube_CECWeld_FOS, Port_Circum_HEC, Port_Circum_CEC
#         ],
#         "Desired Value": [
#             4, 7, 4, 4, 4, 7, 4, 7, 4, 2, 4, 7, 4, 7, 7, 2, 2
#         ]
#     }
#     fos_df = pd.DataFrame(fos_summary)

#     # Create Excel in memory
#     excel_buffer = io.BytesIO()
#     with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
#         workbook = writer.book

#         # --- Sheet 1: FOS Summary ---
#         fos_df.to_excel(writer, sheet_name="FOS Summary", index=False)
#         fos_ws = writer.sheets["FOS Summary"]
#         header_fmt = workbook.add_format({'bold': True, 'bg_color': '#D9E1F2'})
#         for col_num, value in enumerate(fos_df.columns.values):
#             fos_ws.write(0, col_num, value, header_fmt)

#         # --- Sheet 2: Cylinder Details ---
#         cylinder_df.to_excel(writer, sheet_name="Cylinder Details", index=False)
#         cyl_ws = writer.sheets["Cylinder Details"]
#         cyl_ws.write(0, 0, "Cylinder Details", header_fmt)

#         # --- Sheet 3: Nearest Neighbors ---
#         try:
#             selected_cols = ['Component', 'Cushioning', 'Working Pressure', 'STROKE', 'BORE', 'ROD DIA', 'Test_Pressure', 'Target', 'Results', 'Part_Failed', 'Completed_Cycles', 'Report_No']
#             neighbor_df_final = neighbor_data[selected_cols] if all(col in neighbor_data.columns for col in selected_cols) else neighbor_data
#         except Exception:
#             neighbor_df_final = pd.DataFrame({'Info': ['No neighbor data found. Run prediction first.']})
#         neighbor_df_final.to_excel(writer, sheet_name="Passed-Failed Neighbors", index=False)
#         neigh_ws = writer.sheets["Passed-Failed Neighbors"]
#         neigh_ws.write(0, 0, "Nearest Neighbors", header_fmt)

#     excel_buffer.seek(0)

#     # --- Download Button ---
#     st.download_button(
#         label="📥 Download Full Excel Report",
#         data=excel_buffer,
#         file_name="Cylinder_FOS_Report.xlsx",
#         mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
#     )

# Button 1: Predict Result
if st.button("🔍 Predict Result"):
    st.subheader("Prediction")

    # Preprocess input for prediction
    input_df = pd.DataFrame([input_dict])
    input_cat = ohe.transform(input_df[cat_cols])
    cat_feature_names = ohe.get_feature_names_out(cat_cols)
    X_cat_df = pd.DataFrame(input_cat, columns=cat_feature_names, index=input_df.index)
    X_cat_df = X_cat_df.drop(columns=[cat_feature_names[0], cat_feature_names[-1]])

    input_num = scaler.transform(input_df[num_cols])
    X_num_df = pd.DataFrame(input_num, columns=num_cols, index=input_df.index)
    input_processed = pd.concat([X_num_df, X_cat_df], axis=1)
    for col in nn_features:
        if col not in input_processed.columns:
            input_processed[col] = 0
    input_processed = input_processed[nn_features]

    # Predict
    nn_prediction = nn_model.predict(input_processed)[0]
    pred_label = "Failed" if nn_prediction == 1 else "Passed"
    result_color = "red" if pred_label == "Failed" else "green"

    # Show UI result
    st.markdown(f"""
    <div style="padding: 30px; border-radius: 10px; background-color: {result_color}; color: white; text-align: center;">
        <h2>{pred_label}</h2>
    </div>
    """, unsafe_allow_html=True)

    # Save prediction in session
    st.session_state.prediction = pred_label

    # Compute distances and neighbors
    from sklearn.metrics.pairwise import euclidean_distances
    distances = euclidean_distances(X_processed[nn_features], input_processed.values).flatten()
    top_k_indices = distances.argsort()[:k]
    neighbor_df = data.iloc[top_k_indices]

    selected_cols = ['Component', 'Cushioning', 'Working Pressure', 'STROKE', 'BORE', 'ROD DIA', 'Test_Pressure',
                     'Target', 'Results', 'Part_Failed', 'Completed_Cycles', 'Report_No']
    neighbor_df_final = neighbor_df[selected_cols] if all(col in neighbor_df.columns for col in selected_cols) else neighbor_df

    # Show in UI
    tab1, tab2 = st.tabs(["✅ Passed", "❌ Failed"])
    with tab1:
        passed = neighbor_df_final[neighbor_df_final['Results'] == 'Passed']
        st.write(f"**{len(passed)} Passed Neighbors:**")
        st.dataframe(passed)

    with tab2:
        failed = neighbor_df_final[neighbor_df_final['Results'] == 'Failed']
        st.write(f"**{len(failed)} Failed Neighbors:**")
        st.dataframe(failed)

    # Save neighbors and input for export
    st.session_state.input_dict = input_dict.copy()
    st.session_state.fos_summary = {
        "Parameter": [
            "Rod_Axial_FOS", "Rod_Shear_FOS", "Rod_Buckling_FOS",
            "Eye_Thickness_FOS", "Leasteye_Thickness_FOS", "Eye_Shear_FOS",
            "Piston_Axial_FOS", "Piston_Shear_FOS",
            "Tube_Axial_FOS", "Tube_Circum_FOS",
            "CEC_Tensile_FOS", "CEC_Shear_FOS",
            "HEC_Tensile_FOS", "HEC_Shear_FOS",
            "Tube_CECWeld_FOS", "HECsideHole_Circum_FOS", "CECsideHole_Circum_FOS"
        ],
        "Calculated Value": [
            Rod_Axial_FOS, Rod_Shear_FOS, Rod_Buckling_FOS,
            Eye_Thickness_FOS, Least_Eye_FOS, Eye_Shear_FOS,
            Piston_Axial_FOS, Piston_Shear_FOS,
            Tube_Axial_FOS, Tube_Circum_FOS,
            CEC_Tensile_FOS, CEC_Shear_FOS,
            HEC_Tensile_FOS, HEC_Shear_FOS,
            Tube_CECWeld_FOS, Port_Circum_HEC, Port_Circum_CEC
        ],
        "Desired Value": [
            4, 7, 4, 4, 4, 7, 4, 7, 4, 2, 4, 7, 4, 7, 7, 2, 2
        ]
    }
    st.session_state.neighbor_df = neighbor_df_final

if 'prediction' in st.session_state:
    if st.button("📤 Export Report to Excel"):
        input_dict_export = st.session_state.input_dict.copy()
        input_dict_export["Prediction_Result"] = st.session_state.prediction
        cylinder_df = pd.DataFrame(list(input_dict_export.items()), columns=["Parameter", "Value"])

        fos_df = pd.DataFrame(st.session_state.fos_summary)
        neighbor_df = st.session_state.neighbor_df.copy()

        # Excel file
        import io
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            workbook = writer.book
            header_fmt = workbook.add_format({'bold': True, 'bg_color': '#D9E1F2'})

            # FOS Summary
            fos_df.to_excel(writer, sheet_name="FOS Summary", index=False)
            for col_num, value in enumerate(fos_df.columns.values):
                writer.sheets["FOS Summary"].write(0, col_num, value, header_fmt)

            # Cylinder Details
            cylinder_df.to_excel(writer, sheet_name="Cylinder Details", index=False)
            writer.sheets["Cylinder Details"].write(0, 0, "Cylinder Details", header_fmt)

            # Nearest Neighbors
            neighbor_df.to_excel(writer, sheet_name="Passed-Failed Neighbors", index=False)
            writer.sheets["Passed-Failed Neighbors"].write(0, 0, "Nearest Neighbors", header_fmt)

        excel_buffer.seek(0)
        st.download_button(
            label="📥 Download Full Excel Report",
            data=excel_buffer,
            file_name="Cylinder_FOS_Report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
else:
    st.warning("")



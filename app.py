from flask import Flask, request, render_template, redirect, url_for, session, send_file, jsonify
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

app.secret_key = '08060ea2f7726d1d6d638d15a8091b45462c829c863e864ee2e1fceb9b4a204d'

# Load Models
pd_model = joblib.load("C:/Users/User/OneDrive/Desktop/Parkinson/model/detection_model.pkl")
severity_model = joblib.load("C:/Users/User/OneDrive/Desktop/Parkinson/model/severity (1).pkl")
fast_prog = joblib.load("C:/Users/User/OneDrive/Desktop/Parkinson/model/future_progression_risk.pkl")
fallout_model = joblib.load("C:/Users/User/OneDrive/Desktop/Parkinson/model/fall_risk.pkl")
subtype_model = joblib.load("C:/Users/User/OneDrive/Desktop/Parkinson/model/parkinson_subtype_model (6).pkl")
QoL = joblib.load("C:/Users/User/OneDrive/Desktop/Parkinson/model/QoL.pkl")
scaler_full = joblib.load("C:/Users/User/OneDrive/Desktop/Parkinson/model/scaler.pkl")





# Define Mappings
severity_mapping = {0: "Mild", 1: "Mild",2 : "Mild" , 
                    3:"Early-Stage",4:"Early-Stage",5:"Mid-Stage",6:"Mid-Stage",
                    7:"Advanced", 8:"Advanced", 9:"Severe", 10:"Severe"}

fallout_mapping = {0: "Low Fall Risk",  1: "High Fall Risk"}

risk_mapping = { 0: "Stable", 1: "High Risk" }

qol_mapping = {
    "Low": "Low QoL (Needs Immediate Support)",
    "Medium": "Moderate QoL (Manageable but Needs Monitoring)",
    "High": "High QoL (Well Maintained)"
}

subtype_mapping = {
    0: "Tremor-Dominant (TD) - Characterized by prominent tremors with minimal rigidity or bradykinesia.",
    1: "Voice-Dominant (VD) - Primarily affects speech, voice modulation, and vocal strength.",
    2: "Mixed-Type (MT) - A combination of tremors, rigidity, and speech issues."
   }

therapy_recommendations = {
    "Mild": [
        "Speech Therapy: Gentle vocal exercises to improve breath control and voice clarity, focusing on volume and articulation.",
        "Occupational Therapy: Training in fine motor skills to maintain daily living activities; ergonomic tools for writing and eating.",
        "Motor Exercises: Light aerobic activities such as walking, stretching, and beginner yoga to maintain flexibility and coordination.",
        "Cognitive Therapy: Memory exercises and problem-solving tasks to support mental agility and delay cognitive decline.",
        "Postural Training: Gentle balance exercises to prevent early postural instability.",
        "Education & Support: Counseling for patient and family to understand symptoms and promote adherence to therapy."
    ],

    "Early-Stage": [
        "Speech Therapy: More structured sessions emphasizing speech volume, pacing, and clarity.",
        "Occupational Therapy: Adaptive strategies and utensils to maintain independence in self-care and household tasks.",
        "Balance & Strength Training: Incorporate Tai Chi and resistance band exercises to improve muscular strength and stability.",
        "Gait Training: Practice large stepping and rhythmic walking; use of metronome or auditory cues recommended.",
        "Physiotherapy: Regular stretching and joint mobility exercises to reduce stiffness.",
        "Psychological Support: Techniques like mindfulness and group therapy to manage anxiety or depression."
    ],

    "Mid-Stage": [
        "Speech Therapy: Intensive vocal and swallowing exercises to counteract dysphagia and speech difficulties.",
        "Occupational Therapy: Training in use of assistive devices (e.g., grab bars, walkers) and energy conservation techniques.",
        "Physiotherapy: Targeted exercises to maintain range of motion and prevent contractures.",
        "Balance & Gait Rehabilitation: Use of physical aids, treadmill training, and fall prevention strategies.",
        "Cognitive Stimulation: Structured activities to maintain executive function and delay dementia progression.",
        "Emotional Support: Psychotherapy and possibly music or art therapy to enhance mood and motivation."
    ],

    "Advanced": [
        "Speech Therapy: Use of augmentative and alternative communication devices if speech is severely impaired.",
        "Physiotherapy: Daily passive and active range of motion exercises to maintain joint health.",
        "Occupational Therapy: Home environment modifications for safety, caregiver training for assistance with ADLs.",
        "Swallowing Therapy: Strategies and exercises to manage dysphagia, reduce aspiration risk.",
        "Psychological & Social Support: Regular counseling to support emotional wellbeing of patient and family.",
        "Palliative Care Coordination: Focus on comfort, symptom management, and quality of life."
    ],

    "Severe": [
        "Speech Therapy: Implementation of communication aids and alternative methods for interaction.",
        "Physiotherapy: Continuous joint mobilization and prevention of pressure sores.",
        "Occupational Therapy: Full caregiver assistance with feeding, dressing, hygiene; use of specialized equipment.",
        "Swallowing Management: Pureed diets, thickened liquids, feeding tubes if necessary to maintain nutrition.",
        "Psychological Care: Intensive support to manage depression, anxiety, and end-of-life concerns.",
        "Multidisciplinary Approach: Coordination among neurologists, therapists, dietitians, and palliative care teams."
    ]
}

diet_recommendations = {
    "Mild": [
        "Focus on Antioxidant-Rich Foods: Berries, dark leafy greens, and green tea to combat oxidative stress.",
        "Include Omega-3 Fatty Acids: Salmon, flaxseeds, chia seeds to support brain health and reduce inflammation.",
        "Maintain Adequate Hydration: Aim for 8-10 glasses of water daily to aid digestion and medication efficacy.",
        "Ensure Adequate Vitamin D & Calcium: Dairy products, fortified plant milks, eggs, and mushrooms to support bone health.",
        "Limit Processed Foods: Avoid excessive sugar, trans fats, and high-sodium snacks that can worsen inflammation.",
        "Incorporate Whole Grains: Brown rice, oats, quinoa to promote steady energy levels and digestive health."
    ],

    "Early-Stage": [
        "Increase Fiber Intake: Whole fruits, vegetables, legumes, and oats to reduce constipation, common in Parkinson’s.",
        "Protein Timing: Distribute protein intake evenly throughout the day to avoid interference with levodopa absorption.",
        "Include Magnesium and Vitamin B6 Sources: Nuts, bananas, and whole grains to help reduce muscle cramps and stiffness.",
        "Hydrate Well: Herbal teas and water to maintain hydration and support metabolism.",
        "Avoid Excessive Caffeine and Alcohol: To prevent dehydration and medication interactions.",
        "Choose Soft, Easy-to-Chew Foods: Cooked vegetables, soups, and tender proteins for easier swallowing."
    ],

    "Mid-Stage": [
        "Prioritize Soft, Nutrient-Dense Foods: Mashed vegetables, pureed fruits, well-cooked grains for easier digestion.",
        "Monitor Protein Intake Carefully: Balance protein sources and timing to optimize medication effectiveness.",
        "Increase Omega-3 and Antioxidant Intake: Fatty fish, nuts, seeds, and colorful vegetables to support neural health.",
        "Ensure Adequate Hydration: Fluids should be consumed regularly; consider thickened liquids if swallowing difficulty exists.",
        "Avoid Hard, Dry, or Sticky Foods: Such as nuts, tough meats, and raw vegetables to reduce choking risk.",
        "Small, Frequent Meals: To prevent fatigue and support nutrient absorption."
    ],

    "Advanced": [
        "Adopt Pureed and Soft Diet: Smooth purees, soft scrambled eggs, yogurt, and thickened liquids to ensure safe swallowing.",
        "Increase Caloric Density: Use nutrient-rich smoothies and supplements to prevent weight loss and muscle wasting.",
        "Maintain Hydration with Modified Liquids: Thickened water and herbal teas to reduce aspiration risk.",
        "Avoid Foods that Cause Choking or Are Difficult to Swallow: Raw vegetables, nuts, tough meats, and sticky foods.",
        "Vitamin and Mineral Supplementation: B12, calcium, and iron as needed, guided by blood tests.",
        "Small, Frequent Meals: To reduce fatigue and improve digestion."
    ],

    "Severe": [
        "Rely on Texture-Modified Diets: Pureed, mechanically soft, and thickened liquids to ensure safe swallowing and reduce aspiration.",
        "Frequent, Small Meals: To maximize nutrient intake without overwhelming swallowing ability.",
        "High-Calorie, High-Protein Supplements: Use medically formulated nutrition shakes to maintain energy and muscle mass.",
        "Avoid All Hard, Crunchy, or Sticky Foods: Including nuts, seeds, raw vegetables, tough meats, and dry bread.",
        "Close Monitoring of Hydration and Nutrition Status: Work closely with healthcare providers to adjust feeding methods.",
        "Consider Enteral Feeding (if necessary): PEG tube feeding may be required to maintain adequate nutrition and hydration."
    ]
}



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        full_feature_order = [
            'Jitter (%)', 'Shimmer (%)', 'Harmonics-to-Noise Ratio (HNR)',
            'Writing Speed (mm/s)', 'Loop Closure Error (%)',
            'Tremor Intensity', 'Number of Pauses',
            'Tremor Frequency (Hz)', 'Bradykinesia Score'
        ]

        # Validate all inputs exist and convert to float
        input_vals = []
        for feature in full_feature_order:
            val = data.get(feature)
            if val is None:
                return f"Error: Missing input for {feature}", 400
            try:
                input_vals.append(float(val))
            except ValueError:
                return f"Error: Invalid input for {feature}", 400
        
        input_data = np.array(input_vals).reshape(1, -1)
        scaled_data = scaler_full.transform(input_data)
       
        
       
        pd_prediction = int(pd_model.predict(scaled_data)[0])
        pd_result = "Parkinson’s Detected" if pd_prediction == 1 else "No Parkinson’s"
        
        
        if pd_result == "Parkinson’s Detected":

            

           severity_pred = int(severity_model.predict(scaled_data)[0])
           severity_result = severity_mapping[severity_pred]
        
       
           fallout_pred = fallout_model.predict(scaled_data)[0]
           fallout_result = fallout_mapping[int(fallout_pred)]

           subtype_pred = int(subtype_model.predict(scaled_data)[0])
           subtype_result = subtype_mapping[subtype_pred]
        
           progression_pred = int(fast_prog.predict(scaled_data)[0])
           progression_result = risk_mapping[progression_pred]
        
        
           qol_pred = QoL.predict(scaled_data)[0]  
           qol_result = qol_mapping.get(qol_pred, "Unknown QoL Category")


         


               # ✅ Store results in session
           session['results'] = {
            "PD Diagnosis": pd_result,
            "Severity Stage": severity_result,
            "Fall-Out Risk": fallout_result,
            "Future Progression Risk" : progression_result,
            "Subtype": subtype_result,
            "Therapy Recommendation": therapy_recommendations[severity_result],
            "Diet Recommendation": diet_recommendations[severity_result],
            "Quality of Life" : qol_result,
            "Consult Neurologist": "Yes" if severity_result in ["Mid-Stage", "Advanced", "Severe"] or fallout_result == "High Fall Risk" else "No",
            
           }

        else:
            session['results'] = {
    "PD Diagnosis": pd_result,
    "Severity Stage": "Not Applicable – Patient is Healthy",
    "Fall-Out Risk": "Negligible – No Risk Identified",
    "Future Progression Risk": "No Progression Expected – Normal Neurological Profile",
    "Subtype": "Not Applicable – No Parkinson’s Diagnosis",
    "Therapy Recommendation": [
        "No therapy required at this time.",
        "Continue regular physical activity and brain exercises.",
        "Schedule periodic health check-ups to monitor overall well-being."
    ],
    "Diet Recommendation": [
        "Maintain a balanced and nutritious diet.",
        "Include foods rich in antioxidants and omega-3 for brain health.",
        "Stay hydrated and avoid excessive processed foods."
    ],
    "Quality of Life": "Excellent",
    "Consult Neurologist": "No"
    
    }

        return redirect(url_for('results'))
    
    except Exception as e:
        return f"Error: {e}", 400  # Return error if something is wrong




@app.route('/results')
def results():
    if 'results' not in session:
        return redirect(url_for('index'))
    return render_template('starter-page.html', results=session['results'])

@app.route('/download_report')
def download_report():
    if 'results' not in session:
        return redirect(url_for('index'))

    data = session['results']
    report_content = f"""
    Parkinson’s Disease Prediction Report
    -------------------------------------
    - PD Diagnosis: {data['PD Diagnosis']}
    - Severity Stage: {data['Severity Stage']}
    - Fall-Out Risk: {data['Fall-Out Risk']}
    - Subtype: {data['Subtype']}
    - Progression Risk: {data['Future Progression Risk']}
    - Quality of Life: {data['Quality of Life']}
    - Therapy Recommendation: {data['Therapy Recommendation']}
    - Diet Recommendation: {data['Diet Recommendation']}
    - Consult Neurologist: {data['Consult Neurologist']}
    """

    report_path = "static/patient_report.txt"
    with open(report_path, "w") as file:
        file.write(report_content)

    return send_file(report_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)











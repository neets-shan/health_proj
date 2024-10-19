from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import seaborn as sns

app = Flask(__name__)

# Load data function
def load_data():
    try:
        df = pd.read_csv('healthdata.csv')
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()
    return df

df = load_data()

# Function to calculate CVH
def calculate_cvh(age, bmi, smoking, physical_activity, blood_pressure, cholesterol, fasting_blood_sugar):
    score = 0
    score += 2 if smoking == '0' else 0
    score += 2 if bmi < 25 else (1 if 25 <= bmi < 30 else 0)
    score += 2 if physical_activity == '2' else (1 if physical_activity == '1' else 0)
    score += 2 if blood_pressure == '0' else (1 if blood_pressure == '1' else 0)
    score += 2 if cholesterol == '0' else (1 if cholesterol == '1' else 0)
    score += 2 if fasting_blood_sugar == '0' else (1 if fasting_blood_sugar == '1' else 0)
    
    if score >= 10:
        return 0  # Ideal
    elif 5 <= score < 10:
        return 1  # Intermediate
    else:
        return 2  # Poor

# Ensure numeric columns are correctly set
numeric_cols = ['age', 'bmi', 'blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'healthcare_cost']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

@app.route('/', methods=['GET', 'POST'])
def index():
    cvh_category = None
    user_bmi = None
    correlation = None
    suggestions = ""
    cvh_numeric = None  # Initialize to avoid UnboundLocalError

    if request.method == 'POST':
        try:
            age = float(request.form['age'])
            bmi = float(request.form['bmi'])
            smoking = request.form['smoking']
            physical_activity = request.form['physical_activity']
            blood_pressure = request.form['blood_pressure']
            cholesterol = request.form['cholesterol']
            glucose = request.form['glucose']
        except ValueError as e:
            print(f"Error converting form input: {e}")
            return "Invalid input. Please ensure all values are numeric."

        # Call CVH function
        cvh_numeric = calculate_cvh(age, bmi, smoking, physical_activity, blood_pressure, cholesterol, glucose)
        cvh_category = ['Ideal', 'Intermediate', 'Poor'][cvh_numeric]
        user_bmi = bmi

        # Add user's data to the dataframe
        user_data = pd.DataFrame({
            'age': [age],
            'bmi': [bmi],
            'smoking': [smoking],
            'physical_activity': [physical_activity],
            'blood_pressure': [blood_pressure],
            'cholesterol': [cholesterol],
            'fasting_blood_sugar': [glucose],
            'cvh_category': [cvh_numeric],
            'healthcare_cost': [0]  # Assuming healthcare cost is zero initially
        })

        global df
        df = pd.concat([df, user_data], ignore_index=True)

        # Calculate correlation
        correlation = df['healthcare_cost'].corr(df['cvh_category'])

        # Suggestion based on CVH category
        if cvh_numeric == 1:  # Intermediate
            suggestions = "To improve your CVH category, consider increasing your physical activity, maintaining a healthy diet, and managing your weight."
        elif cvh_numeric == 2:  # Poor
            suggestions = "To improve your CVH category, it's essential to quit smoking, adopt a balanced diet rich in fruits and vegetables, exercise regularly, and monitor your health."

    # Generate graphs only if user_bmi is defined
    img_bar_cvh = generate_bmi_comparison_plot(user_bmi) if user_bmi is not None else None
    img_box_healthcare = generate_box_plot_healthcare_costs()
    img_linear_healthcare_cost = generate_linear_healthcare_cost_plot()

    # Only generate correlation plot if cvh_numeric has a valid value
    img_correlation = generate_correlation_plot(user_bmi, cvh_numeric) if cvh_numeric is not None else None

    return render_template('index.html', 
                           result=cvh_category, 
                           bar_graph_cvh=img_bar_cvh, 
                           box_graph_healthcare=img_box_healthcare, 
                           linear_healthcare_cost_plot=img_linear_healthcare_cost,
                           correlation=correlation,
                           correlation_graph=img_correlation,
                           suggestions=suggestions)

# Function to generate BMI comparison plot
def generate_bmi_comparison_plot(user_bmi):
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='bmi', kde=True)
    plt.axvline(user_bmi, color='red', linestyle='--', label='User BMI')
    plt.title('BMI Comparison')
    plt.xlabel('BMI')
    plt.ylabel('Frequency')
    plt.legend()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()  # Close the plot to prevent display issues
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf8')

# Function to generate healthcare costs box plot
def generate_box_plot_healthcare_costs():
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='cvh_category', y='healthcare_cost', data=df, palette='Set2')
    plt.title('Healthcare Costs by CVH Category')
    plt.ylabel('Healthcare Costs')
    plt.xlabel('CVH Category')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()  # Close the plot to prevent display issues
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf8')

# Function to generate linear healthcare cost plot
def generate_linear_healthcare_cost_plot():
    plt.figure(figsize=(8, 5))
    avg_costs = df.groupby('cvh_category')['healthcare_cost'].mean().reset_index()
    plt.plot(avg_costs['cvh_category'], avg_costs['healthcare_cost'], marker='o', color='orange')
    plt.title('Average Healthcare Costs by CVH Category')
    plt.xlabel('CVH Category')
    plt.ylabel('Average Healthcare Cost')
    plt.xticks([0, 1, 2], ['Ideal', 'Intermediate', 'Poor'])
    plt.grid()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()  # Close the plot to prevent display issues
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf8')

# Function to generate correlation plot
def generate_correlation_plot(user_bmi, cvh_numeric):
    plt.figure(figsize=(8, 5))
    plt.scatter(df['cvh_category'], df['healthcare_cost'], color='blue', label='Existing Data')
    plt.scatter(cvh_numeric, 0, color='red', label='User Input', s=100)  # User input point
    plt.title('Correlation between CVH Category and Healthcare Cost')
    plt.xlabel('CVH Category')
    plt.ylabel('Healthcare Cost')
    plt.xticks([0, 1, 2], ['Ideal', 'Intermediate', 'Poor'])
    plt.legend()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()  # Close the plot to prevent display issues
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf8')

if __name__ == '__main__':
    app.run(debug=True)

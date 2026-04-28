import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Machine Learning App",
    layout="wide"
)

@st.cache_data
def load_class_results():
    return pd.read_csv("results/tuned_models_results.csv")

@st.cache_resource
def load_class_models():
    models = {
        "CatBoost optimizat": joblib.load("models/catboost_model.pkl"),
        "Explainable Boosting Machine optimizat": joblib.load("models/ebm_model.pkl"),
        "XGBoost optimizat": joblib.load("models/xgboost_model.pkl"),
        "Random Forest optimizat": joblib.load("models/random_forest_model.pkl"),
        "Decision Tree optimizat": joblib.load("models/decision_tree_model.pkl")
    }

    return models

@st.cache_data
def load_class_data():
    return pd.read_csv("data/classification_dataset.csv")

page = st.sidebar.radio(
    "Alege pagina:",
    ["Clasificare", "Regresie"]
)

if page == "Clasificare":
    st.title("Clasificare")

    #INTRODUCERE
    st.write("""
            Această pagină este dedicată problemei de clasificare binară.
    
            Modelul trebuie să prezică variabila 'HiringDecision', unde:
                - '1' înseamnă candidat angajat;
                - '0' înseamnă candidat neangajat.
    """)

    #INCARCARE MODELE SI REZULTATE
    st.markdown("## Modele optimizate pentru clasificare")

    classification_results = load_class_results()
    classification_models = load_class_models()
    classification_data = load_class_data()

    st.write("Tabelul de mai jos prezintă performanțele celor 5 modele optimizate.")

    st.dataframe(classification_results)

    #SELECTAREA MODELULUI
    st.markdown("## Selectarea modelului")

    selected_model_name = st.selectbox(
        "Alege modelul pentru test:",
        classification_results["Model"].tolist()
    )

    selected_model = classification_models[selected_model_name]

    #AFISARE METRICI MODEL SELECTAT
    st.markdown("### Metricile modelului selectat")

    selected_model_results = classification_results[
        classification_results["Model"] == selected_model_name
    ]

    row = selected_model_results.iloc[0]
    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Accuracy", f"{row['Accuracy']:.4f}")
    col2.metric("Precision", f"{row['Precision']:.4f}")
    col3.metric("Recall", f"{row['Recall']:.4f}")
    col4.metric("F1-score", f"{row['F1-score']:.4f}")
    col5.metric("ROC-AUC", f"{row['ROC-AUC']:.4f}")

    #INTRODUCERE DATE TEST
    st.markdown("## Introducerea datelor candidatului")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=60,
            value=30
        )

        gender = st.selectbox(
            "Gender",
            options=[0, 1],
            format_func=lambda x: "0 - Female" if x == 0 else "1 - Male"
        )

        education_level = st.selectbox(
            "EducationLevel",
            options=[1, 2, 3, 4]
        )

        experience_years = st.number_input(
            "ExperienceYears",
            min_value=0,
            max_value=20,
            value=5
        )

        previous_companies = st.number_input(
            "PreviousCompanies",
            min_value=0,
            max_value=10,
            value=2
        )

    with col2:
        distance_from_company = st.number_input(
            "DistanceFromCompany",
            min_value=0.0,
            max_value=100.0,
            value=25.0
        )

        interview_score = st.number_input(
            "InterviewScore",
            min_value=0,
            max_value=100,
            value=70
        )

        skill_score = st.number_input(
            "SkillScore",
            min_value=0,
            max_value=100,
            value=70
        )

        personality_score = st.number_input(
            "PersonalityScore",
            min_value=0,
            max_value=100,
            value=70
        )

        recruitment_strategy = st.selectbox(
            "RecruitmentStrategy",
            options=[1, 2, 3]
        )

    input_data = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "EducationLevel": education_level,
        "ExperienceYears": experience_years,
        "PreviousCompanies": previous_companies,
        "DistanceFromCompany": distance_from_company,
        "InterviewScore": interview_score,
        "SkillScore": skill_score,
        "PersonalityScore": personality_score,
        "RecruitmentStrategy": recruitment_strategy
    }])

    #PREEDICTIE
    if st.button("Generează predicția"):
        prediction = selected_model.predict(input_data)[0]
        prediction_proba = selected_model.predict_proba(input_data)[0]

        st.markdown("## Rezultatul predicției")

        if prediction == 1:
            st.success("Candidatul este angajat.")

            st.metric(
                "Probabilitate angajat",
                f"{prediction_proba[1]:.4f}"
            )

        else:
            st.error("Candidatul nu este angajat.")

            st.metric(
                "Probabilitate neangajat",
                f"{prediction_proba[0]:.4f}"
            )

        #ANALIZA SHAP PENTRU CATBOOST
        st.markdown("## Explicație SHAP pentru predicția generată")

        if selected_model_name == "CatBoost optimizat":
            X_background = classification_data.drop(columns=["HiringDecision"])

            explainer = shap.Explainer(selected_model, X_background)
            shap_values = explainer(input_data)

            shap_df = pd.DataFrame({
                "Caracteristică": input_data.columns,
                "Valoare introdusă": input_data.iloc[0].values,
                "Valoare SHAP": shap_values.values[0]
            })

            shap_df["Influență"] = shap_df["Valoare SHAP"].abs()
            shap_df = shap_df.sort_values(by="Influență", ascending=False)

            st.write("""
                    Valorile SHAP arată cât de mult a influențat fiecare caracteristică predicția.
                    O valoare SHAP pozitivă împinge predicția spre clasa '1' - angajat,
                    iar o valoare SHAP negativă împinge predicția spre clasa '0' - neangajat.
            """)

            st.dataframe(shap_df)

            st.markdown("### Grafic SHAP pentru predicția curentă")

            fig, ax = plt.subplots(figsize=(8, 5))

            ax.barh(
                shap_df["Caracteristică"],
                shap_df["Valoare SHAP"]
            )

            ax.set_xlabel("Valoare SHAP")
            ax.set_ylabel("Caracteristică") 
            ax.set_title("Influența caracteristicilor asupra predicției")

            st.pyplot(fig)

        else:
            st.info("""
                    Explicația SHAP e disponibilă doar pentru cel mai performant model.
            """)
    
    #HIPERPARAMETRI
    st.markdown("## Hiperparametrii modelului selectat")

    params = selected_model.get_params()

    params_df = pd.DataFrame(
        list(params.items()),
        columns=["Hiperparametru", "Valoare"]
    )

    st.dataframe(params_df)
    
    #GRAFICE EDA
    st.markdown("## Grafice EDA relevante")

    st.write("""
            Mai jos sunt prezentate câteva grafice obținute în etapa de analiză exploratorie a datelor.
            Acestea ajută la înțelegerea distribuției variabilei țintă și a scorurilor candidaților.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            "results/eda_hiring_decision.png",
            use_container_width=True
        )

        st.image(
            "results/eda_interview_score.png",
            use_container_width=True
        )

    with col2:
        st.image(
            "results/eda_skill_score.png",
            use_container_width=True
        )

        st.image(
            "results/eda_personality_score.png",
            use_container_width=True
        )

    #CURBE DE INVATARE
    st.markdown("## Curba de învățare pentru modelul selectat")

    learning_curve_paths = {
        "CatBoost optimizat": "results/learning_curve_cat.png",
        "Explainable Boosting Machine optimizat": "results/learning_curve_ebm.png",
        "XGBoost optimizat": "results/learning_curve_xgb.png",
        "Random Forest optimizat": "results/learning_curve_rf.png",
        "Decision Tree optimizat": "results/learning_curve_dt.png"
    }

    st.write("""
    Curba de învățare arată evoluția scorului pe datele de antrenare și pe datele de validare,
    în funcție de numărul de exemple folosite la antrenare.
    """)

    st.image(
        learning_curve_paths[selected_model_name],
        width=700
    )


elif page == "Regresie":
    st.title("Regresie")

    st.write("""
            Această pagină este dedicată problemei de regresie.
    """)
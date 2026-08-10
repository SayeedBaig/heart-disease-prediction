from digital_twin.models.patient_profile import PatientProfile


patients = [

    PatientProfile(

        patient_id="P001",

        age=55,
        gender="Male",

        systolic_bp=150,
        diastolic_bp=95,

        cholesterol=260,
        glucose=170,

        bmi=32,
        weight=85,

        smoking_status=True,

        alcohol_consumption=False,

        exercise_level="Low",

        fusion_risk_percentage=82

    ),

    PatientProfile(

        patient_id="P002",

        age=60,
        gender="Female",

        systolic_bp=None,
        diastolic_bp=90,

        cholesterol=240,
        glucose=140,

        bmi=30,
        weight=78,

        smoking_status=False,

        alcohol_consumption=False,

        exercise_level="Moderate",

        fusion_risk_percentage=74

    ),

    PatientProfile(

        patient_id="P003",

        age=85,
        gender="Male",

        systolic_bp=220,
        diastolic_bp=120,

        cholesterol=360,
        glucose=250,

        bmi=42,
        weight=110,

        smoking_status=True,

        alcohol_consumption=True,

        exercise_level="Low",

        fusion_risk_percentage=95

    ),

    PatientProfile(

        patient_id="P004",

        age=25,
        gender="Female",

        systolic_bp=118,
        diastolic_bp=75,

        cholesterol=170,
        glucose=90,

        bmi=22,
        weight=58,

        smoking_status=False,

        alcohol_consumption=False,

        exercise_level="High",

        fusion_risk_percentage=35

    )

]
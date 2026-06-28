# Digital Twin Design Document

## 1. Introduction

### What is a Digital Twin?

A Digital Twin is a virtual representation of a patient that can simulate future health outcomes based on current medical information. In this project, the Digital Twin will use outputs from the Clinical, ECG, Echo, and Fusion modules to estimate how a patient's cardiovascular risk may change over time under different intervention scenarios.

### Purpose

The Digital Twin module is designed to:

* Simulate future cardiovascular risk progression.
* Evaluate the impact of lifestyle and medical interventions.
* Support personalized risk assessment.
* Provide future risk projections without modifying the existing prediction pipeline.
* Extend the functionality of the current multimodal heart disease prediction system.

---

## 2. Objectives

The Digital Twin module aims to:

1. Predict future cardiovascular risk using existing model outputs.
2. Simulate different intervention scenarios.
3. Compare projected outcomes across multiple scenarios.
4. Support explainable and personalized healthcare recommendations.
5. Integrate seamlessly with the existing Clinical, ECG, Echo, and Fusion modules.

---

## 3. System Architecture

Current Pipeline:

Clinical Module + ECG Module + Echo Module

↓

Fusion Module

↓

Risk Prediction

Proposed Pipeline:

Clinical Module + ECG Module + Echo Module

↓

Fusion Module

↓

Digital Twin Module

↓

Risk Projection & Intervention Simulation

↓

Visualization / Reporting Layer

The Digital Twin is designed as a downstream component and does not replace any existing prediction module.

---

## 4. Inputs to the Digital Twin

The Digital Twin should use information already produced by the existing system.

### Clinical Module Outputs

* Clinical Risk Level
* Clinical Risk Score

### ECG Module Outputs

* ECG Risk Level
* ECG Risk Score

### Echo Module Outputs

* Echo Risk Level
* Echo Risk Score

### Fusion Module Outputs

* Final Risk Level
* Risk Percentage

### Simulation Parameters

* Simulation Horizon
* Intervention Scenario
* Weight Change
* Blood Pressure Change
* Cholesterol Change
* Glucose Change
* Smoking Status Change
* Physical Activity Change

---

## 5. Risk Progression Model

The repository currently does not contain a dedicated longitudinal cardiovascular progression model. Therefore, the Digital Twin will use a simulation approach that is compatible with the existing implementation.

### Proposed Method

For each future year:

1. Increase patient age.
2. Apply selected intervention changes.
3. Recompute clinical features.
4. Generate updated Clinical predictions.
5. Combine updated Clinical output with baseline ECG and Echo outputs.
6. Recalculate Fusion risk.
7. Store projected risk values.

### Assumptions

* ECG results remain constant unless new ECG data is provided.
* Echo results remain constant unless new Echo data is provided.
* Fusion output remains the primary risk indicator.

---

## 6. Simulation Parameters (examples)

The following variables can be modified during simulation:

### Weight Reduction

* 5% reduction
* 10% reduction
* 15% reduction

### Blood Pressure Improvement

* Reduced systolic pressure
* Reduced diastolic pressure

### Cholesterol Improvement

* Cholesterol category improvement

### Glucose Improvement

* Glucose category improvement

### Lifestyle Changes

* Smoking cessation
* Reduced alcohol consumption
* Increased physical activity

---

## 7. Intervention Scenarios (examples)

### Scenario 1: No Intervention

The patient profile remains unchanged except for age progression.

### Scenario 2: Smoking Cessation

Smoking status changes from smoker to non-smoker.

### Scenario 3: Increased Physical Activity

Physical activity status is improved.

### Scenario 4: Weight Reduction

Body weight is reduced over time.

### Scenario 5: Blood Pressure Control

Blood pressure values are reduced to healthier ranges.

### Scenario 6: Cholesterol Management

Cholesterol levels are improved.

### Scenario 7: Glucose Management

Glucose levels are improved.

### Scenario 8: Combined Lifestyle Intervention

Combines:

* Smoking cessation
* Increased activity
* Weight reduction

### Scenario 9: Combined Medical Intervention

Combines:

* Blood pressure control
* Cholesterol management
* Glucose management

---

## 8. Expected Outputs (examples)

The Digital Twin should generate:

### Risk Projection

* Current Risk
* Year 1 Risk
* Year 2 Risk
* Year 3 Risk
* Year 4 Risk
* Year 5 Risk

### Scenario Comparison

* No Intervention
* Lifestyle Improvement
* Medical Management
* Combined Intervention

### Additional Outputs

* Projected Risk Level
* Risk Percentage
* Simulation Assumptions
* Scenario Summary

---


## 9. Future Visualization

The frontend can display:

* 5-Year Risk Timeline
* Multi-Line Risk Projection Chart
* Scenario Comparison Dashboard
* Best Intervention Highlight
* Interactive Simulation Controls

---

## 10. References

1. World Health Organization (WHO) HEARTS Technical Package.
2. ACC/AHA Guideline on the Primary Prevention of Cardiovascular Disease.
3. Corral-Acero et al., The Digital Twin to Enable the Vision of Precision Cardiology.
4. PTB-XL: A Large Publicly Available Electrocardiography Dataset.
5. American Society of Echocardiography Guidelines.

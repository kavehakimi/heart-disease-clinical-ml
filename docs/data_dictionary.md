| Feature | Clinical meaning | Unit / coding | Feature type | Notes |
|---|---|---|---|---|
|age|age|years|Continuous|---|
|sex|sex|1,0|Binary|1 = male; 0 = female|
|cp|chest pain type|1,2,3,4|Nominal categorical|1: typical angina; 2: atypical angina; 3: non-anginal pain; 4: asymptomatic|
|trestbps|resting blood pressure|mm hg|Continuous|in mm Hg on admission to the hospital|
|chol|serum cholestoral|mg/dl|Continuous|---|
|fbs|fasting blood sugar|1,0|Binary|(fasting blood sugar > 120 mg/dl)(1 = true; 0 = false)|
|restecg|resting electrocardiographic results|0,1,2|Nominal categorical|0: normal; 1: having ST-T wave abnormality; 2: showing probable or definite left ventricular hypertrophy by Estes' criteria|
|thalach|maximum heart rate achieved|---|Continuous|---|
|exang|exercise induced angina|1,0|Binary|1 = yes; 0 = no|
|oldpeak|ST depression induced by exercise relative to rest|numeric value; unit not explicitly specified|Continuous|---|
|slope|the slope of the peak exercise ST segment|1,2,3|Ordinal categorical|1: upsloping; 2: flat; 3: downsloping; Potentially ordinal; encoding choice to be evaluated during preprocessing.|
|ca|number of major vessels|---|Discrete numerical (count)|Expected values 0–3; contains missing values encoded as "?".|
|thal|---|3,6,7|Nominal categorical|3 = normal; 6 = fixed defect; 7 = reversible defect; Contains missing values encoded as "?".|
|num|diagnosis of heart disease (angiographic disease status)|0–4 in the processed dataset|Target|Observed coding: 0–4 in processed.cleveland.data. Documentation defines 0 as <50% diameter narrowing and 1 as >50% diameter narrowing. Values 2–4 are also observed in the processed data; their finer interpretation should not be assumed without additional source verification.
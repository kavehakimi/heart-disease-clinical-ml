# Cleaning Decision Log

This document records data-quality issues identified in the Cleveland
Heart Disease dataset and the decisions made before data cleaning and
model development.

The purpose is to maintain a transparent and reproducible record of
what was changed, retained, or deferred, together with the rationale
for each decision.

## Decision Log

| ID | Issue | Evidence | Decision | Rationale | Status |
| DQ-01 | Missing values in `ca` | 4 values encoded as `"?"`; all other observed values are within the expected 0–3 range. | Use median imputation as the baseline strategy, fitted on training data only after train/test splitting. | Missingness is low (4/303; 1.32%). Complete-case deletion would discard observations from an already small dataset. Mean imputation could create non-integer values inconsistent with the count nature of the feature. More advanced imputation may be evaluated later. | Planned |
| DQ-02 | Missing values in `thal` | 2 values encoded as `"?"`; observed non-missing values are 3, 6 and 7. | Use most-frequent imputation as the baseline strategy, fitted on training data only after train/test splitting. | Missingness is very low (2/303; 0.66%). `thal` is nominal categorical, so numerical mean/median imputation would not appropriately reflect its categorical meaning. Mode imputation provides a simple baseline. | Planned |
| DQ-03 | Duplicate records | No fully duplicated rows were identified among 303 observations. | No action required. | No duplicate records were detected. | Closed |
| DQ-04 | Numerical range validity | Initial range assessment found no values that could be established as impossible or erroneous. | Retain all observations pending further preprocessing. | Extreme values should not be treated as errors without supporting domain evidence. | Closed |
| DQ-05 | IQR candidate outliers in `trestbps` | 9 observations above the IQR upper bound of 170 mmHg; values range from 172 to 200 mmHg. | Retain. | Values are statistically extreme but clinically plausible; no evidence of data-entry error was identified. | Closed |
| DQ-06 | IQR candidate outliers in `chol` | 5 observations above the IQR upper bound of 371 mg/dL: 394, 407, 409, 417 and 564 mg/dL. | Retain. | Values represent very high total cholesterol but are not established as physiologically impossible or erroneous. Statistical extremeness alone does not justify removal. | Closed |
| DQ-07 | IQR candidate outlier in `thalach` | One observation below the IQR lower bound of 84.75 bpm: 71 bpm, observed in a 67-year-old patient. | Retain. | The value is statistically extreme, but no direct evidence indicates measurement or data-entry error. Maximum achieved heart rate is patient-dependent. | Closed |
| DQ-08 | IQR candidate outliers in `oldpeak` | 5 observations above the IQR upper bound of 4.0: 4.2, 4.2, 4.4, 5.6 and 6.2. | Retain; clinical interpretation remains subject to further verification. | Exercise-induced ST depression is clinically meaningful, but the precise interpretation of the `oldpeak` scale requires further verification. No evidence currently indicates data-entry error. | Review if needed |
| DQ-09 | Target coding | Raw `target` contains values 0, 1, 2, 3 and 4. Documentation defines the diagnostic objective in terms of absence/presence of disease, while finer interpretation of observed values 1–4 has not been established from the current documentation. | Preserve the raw target during data cleaning; binary target transformation to be defined explicitly before modelling. | Raw source information should be retained. Target transformation is a modelling/preprocessing decision and should not silently overwrite the original values. | Pending |

## Preprocessing Principles

- Raw source data will not be overwritten.
- Missing-value markers (`"?"`) will be converted to proper missing values (`NaN`) before modelling.
- Any preprocessing step that learns parameters from the data will be fitted on the training set only.
- The test set will not be used to determine imputation values, scaling parameters, feature-selection decisions, or other learned preprocessing parameters.
- Statistical outliers will not be removed solely because they exceed IQR thresholds; removal requires additional evidence of invalidity or data error.
- All target transformations will be explicitly documented and the original target values will be preserved.
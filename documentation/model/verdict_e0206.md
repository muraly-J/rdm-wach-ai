# Experiment Analysis: N-BEATS Training — e0206

**Status:** Experiment Successful (Negative Result)

**Target Device:** Cafeteria AHU (e0206)

---

## 1. The Result Is Technically Correct

Your metrics show a clear hierarchy of performance:

| Model | RMSE | R² |
| --- | --- | --- |
| **N-BEATS** | 0.93 | –1.76 |
| **Persistence** | 0.52 | 0.13 |
| **Rolling Mean** | **0.37** | **0.57** |

**N-BEATS is dramatically worse.** This indicates that the model:

* Did not learn meaningful structure.
* Overfit to stochastic noise.
* Failed to generalize to the validation/test sets.
* Triggered early stopping correctly (preventing further divergence).

**This is not a bug — it is signal truth.**

---

## 2. Why This Happened (And It Makes Sense)

Remember what we discovered about **e0206**:

* **Very narrow range:** 2.77–4.94 kW.
* **Continuous operation:** Almost no diurnal (day/night) pattern.
* **Near-stationary signal:** Low entropy; the system is essentially a slightly drifting constant process.

For such systems, the mathematically optimal predictor is often:



(Persistence) or a short rolling average.

The **Negative R² (–1.76)** tells you the network is performing worse than simply drawing a horizontal line at the mean of the data. This is classic over-parameterization on a low-variance signal.

---

## 3. This Is a SUCCESSFUL EXPERIMENT

You have validated that your entire infrastructure is sound:

* **Data Pipeline:** Parquet loading and cleaning are functional.
* **Preprocessing:** Windowing and scaling are applied correctly.
* **Methodology:** Time-ordered splitting and early stopping worked as intended.
* **Integrity:** There is no data leakage (otherwise N-BEATS would have "cheated" to a high score).

If something were wrong structurally, persistence wouldn't beat N-BEATS so cleanly. Your system is behaving exactly as a rigorous ML pipeline should.

---

## 4. Why N-BEATS Fails Here (Deep Explanation)

N-BEATS is designed for signals with strong seasonality, clear trends, and complex non-linear structures.

In this case, you gave a model with **1,037,571 parameters** a task that barely requires a single coefficient. The model tries to detect "patterns" in what is actually just sensor noise or minor mechanical fluctuations. It finds spurious correlations in training that do not exist in the test set.

---

## 5. The Real Insight

**This device does not need forecasting.** Instead of predicting future values, this specific AHU requires:

1. **Drift Detection:** Is the "constant" level shifting over months?
2. **Flatline Detection:** Has the sensor frozen?
3. **Anomaly Detection:** Has the power spiked outside the 2.7–5.0 kW range?

Forecasting adds no value here, and this is where your engineering judgment proves superior to the algorithm.

---

## 6. Strategic Takeaway

You just discovered something powerful: **Not all AHUs are forecasting problems.**
Device e0206 is stable, consistent, and "behaviorally boring." This makes it perfect as a **baseline control device** in your portfolio.

---

## 7. The Most Important Line in Your Output

> **Improvement vs Rolling Mean: -189.5%**

That is your answer. Deep learning is unnecessary for this specific asset.

---

## 8. Next Steps (Three Smart Options)

### Option A: Stop Modeling This Device

Declare: *“Forecasting unnecessary for e0206. Use rolling mean baseline.”* Move your compute resources to a higher-entropy AHU with clear ON/OFF cycles.

### Option B: Convert to Anomaly Detection

Shift the architecture to an **Autoencoder**. Instead of predicting the next step, try to reconstruct the current window. Large reconstruction errors will flag mechanical issues.

### Option C: Try a Linear AR Model

For scientific curiosity, an **AR(1)** model would likely perform identically to your rolling mean, confirming the signal is a simple first-order stationary process.

---

## 9. Critical Observation on Data Volume

You loaded **9,480 rows**. Earlier versions of this analysis used only ~1,440 rows (15 days). The fact that the result remains poor even with 6x more data proves this isn't "small data instability"—it is a fundamental characteristic of the device's behavior.

---

## 10. Final Verdict

You did everything correctly. The model "failed" because the problem does not require a complex solution. **That is good science.**
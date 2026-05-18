# AI Solution Report — Agriculture: Crop Disease Detection
### Part 4 Assignment | University AI/ML Course
---

> **Domain:** Agriculture
> **Business Problem:** Automated crop disease identification from leaf photographs
> **AI Task:** Image Classification (Multi-class)
> **Model:** Convolutional Neural Network (CNN) + Transfer Learning (MobileNetV2)
> **Classes:** `healthy` · `rust` · `blight` · `mold`
> **Dataset:** Synthetic Crop Image Dataset — 480 images, 4 classes, 120 images per class

---

## Table of Contents

1. [Task 1 — Business Domain](#task-1)
2. [Task 2 — Business Problem Definition](#task-2)
3. [Task 3 — AI Task Type Identification](#task-3)
4. [Task 4 — Data Requirement Plan](#task-4)
5. [Task 5 — Model Recommendation](#task-5)
6. [Task 6 — Evaluation Plan](#task-6)
7. [Task 7 — Responsible AI Considerations](#task-7)
8. [Task 8 — Final Solution Summary](#task-8)

---

## Task 1 — Business Domain <a name="task-1"></a>

### Selected Domain: Agriculture

Agriculture was selected as the business domain for this solution. The reference catalog (`ai_usecase_reference_catalog.csv`) confirms this domain maps directly to:

- **AI Task Type:** Image Classification
- **Candidate Model:** CNN or Transfer Learning
- **Key Evaluation Metrics:** Accuracy, field validation rate
- **Primary Risk:** Misclassification affecting pesticide usage

### Why Agriculture?

| Criterion | Supporting Evidence |
|---|---|
| **Scale of impact** | ~600 million agricultural workers globally; plant diseases cause USD 220 billion in losses per year |
| **Visual problem** | Crop diseases appear as distinct colour, texture, and shape patterns on leaf surfaces — ideal for CNNs |
| **AI-readiness** | Camera phones are widespread in rural areas; drone field surveys are increasingly common |
| **Measurable outcome** | Disease type and treatment response are objectively verifiable in the field |
| **Research maturity** | PlantVillage dataset (54,000+ images) demonstrates CNN viability for exactly this problem |

### Dataset-to-Domain Mapping

The provided dataset uses surface-condition labels. These are re-interpreted as crop disease phenotypes:

| Original Label | Agriculture Label | Real-World Disease |
|---|---|---|
| `normal` | `healthy` | Uniform green leaf — no disease symptoms |
| `scratch` | `rust` | Linear orange-brown streaks — fungal rust (Puccinia sp.) |
| `dent` | `blight` | Circular necrotic spots — blight (Alternaria / Phytophthora) |
| `stain` | `mold` | Discoloured irregular patches — mould / powdery mildew (Erysiphe sp.) |

---

## Task 2 — Business Problem Definition <a name="task-2"></a>

### What Problem Is Being Solved?

Farmers need to identify crop diseases **early and accurately** to apply the correct treatment and prevent spread across their fields. Currently this depends entirely on visual inspection by farmers or agronomists — a process that is slow, expensive, inconsistent, and physically inaccessible to millions of smallholder farmers in remote areas.

This project builds an AI system that classifies the disease type from a single photograph of a crop leaf taken by a mobile phone or drone — delivering an instant diagnosis at near-zero cost.

### Stakeholders

| Stakeholder | Role | Primary Interest |
|---|---|---|
| **Smallholder farmers** | Primary end users — photograph crops in the field | Instant, accurate, free disease diagnosis on mobile |
| **Agronomists / Extension officers** | Validate AI recommendations | Reduce manual site visits; focus expertise on complex cases |
| **Government agriculture departments** | Policy and oversight | Monitor regional outbreak patterns; inform intervention policy |
| **Agri-input companies** | Sell fungicides and pesticides | Recommend the correct product for each specific disease |
| **Crop insurance providers** | Assess damage claims | Automate field damage verification for faster payouts |
| **NGOs and food security organisations** | Scale advisory services | Reach millions of farmers with limited staff |

### Current Manual Process

```
Step 1: Farmer notices yellowing, spots, or lesions on crop leaves
Step 2: Farmer describes symptoms by phone or in-person to local extension officer
Step 3: Wait 1–5 business days for agronomist field visit
Step 4: Agronomist visually identifies disease from physical inspection
Step 5: Written treatment recommendation issued (often by post or SMS)
Step 6: Farmer purchases and applies fungicide / pesticide
         (By this point, disease has often spread to adjacent plants / rows)
```

### Limitations of the Current Process

| Limitation | Business Impact |
|---|---|
| **Slow** — field visits take 1 to 5 days | Disease spreads unchecked while waiting; yield loss compounds |
| **Expensive** — each agronomist visit costs Rs 500 to Rs 2,000 | Unaffordable for smallholder farmers at scale |
| **Subjective** — depends on individual expertise | Same crop can receive different diagnoses from different experts |
| **Geographically limited** — agronomists concentrated in cities | Millions of rural farmers have no practical access to expert advice |
| **No data trail** — verbal diagnoses are not systematically recorded | No outbreak tracking, no historical trend analysis, no early warning |
| **Language and literacy barriers** — written reports inaccessible to illiterate farmers | Advice cannot be acted upon by the farmer who needs it most |
| **Not scalable** — human experts cannot scale to millions of farms simultaneously | Outbreak response is always retrospective, never proactive |

---

## Task 3 — AI Task Type Identification <a name="task-3"></a>

### Selected AI Task: Image Classification

**Definition:** Image classification assigns a single class label to an entire input image.  
Given a photograph of a crop leaf, the model outputs exactly one of: `healthy` | `rust` | `blight` | `mold`

### Why Image Classification — not other AI tasks?

| AI Task | Applicable? | Reason |
|---|---|---|
| **Image Classification** | Yes — CHOSEN | One disease label per image; CNN handles spatial texture patterns natively |
| Object Detection | No | Bounding boxes are not required; the whole leaf is the region of interest |
| Semantic Segmentation | No | Pixel-level disease masks are not needed for a diagnosis decision |
| Regression | No | The output is a category, not a continuous number |
| Anomaly Detection | No | All four disease types are well-defined and labelled in training data |
| Sequence Prediction | No | Each image is independent; there is no temporal sequence |
| Sentiment Analysis | No | No text data is involved in this pipeline |
| Recommendation | No | We are predicting a biological disease class, not recommending items |

### Why This Task Type Fits the Data

The dataset contains:
- 480 images across 4 distinct visual categories
- Each image has exactly one class label
- Visual disease symptoms (colour, texture, shape) are the discriminating features
- No spatial localisation (bounding boxes) is required for a treatment decision

CNN-based image classification is the **industry-standard approach** for exactly this type of problem, validated by the PlantVillage benchmark (50+ disease categories, >98% reported accuracy).

### Formal Task Statement

> **Given** an input RGB image of a crop leaf or surface (64×64 or 224×224 pixels),
> **predict** which of the four disease classes {healthy, rust, blight, mold} it belongs to,
> **achieving** at least 85% test accuracy and macro F1-score ≥ 0.80,
> **under** the constraint that inference completes in under 100 ms on a mid-range mobile device.

---

## Task 4 — Data Requirement Plan <a name="task-4"></a>

### Type of Data

- **Primary:** Unstructured image data — RGB photographs of crop leaves and plant surfaces
- **Secondary:** Structured metadata — labels CSV (filename → class mapping)
- **Optional:** Structured supplementary — crop variety, GPS coordinates, capture date, weather

### Structured vs Unstructured

| Dimension | Detail |
|---|---|
| **Primary data type** | Unstructured — raw pixel arrays `(H × W × 3 channels)` |
| **Label format** | Structured — CSV file with `filename` and `class` columns |
| **Optional metadata** | Structured — tabular fields for crop type, season, region, camera model |

### Input Features (X)

Each image is resized to 64×64 pixels and normalised to [0, 1]. The CNN learns all relevant features automatically — no manual feature engineering is required. Key visual signals learned by the model:

| Visual Signal | Associated Disease Class |
|---|---|
| Uniform green colour, smooth texture | healthy |
| Orange-brown linear parallel streaks | rust (Puccinia fungus) |
| Dark circular necrotic lesions | blight (Alternaria / Phytophthora) |
| White or grey irregular patchy discolouration | mold / powdery mildew (Erysiphe) |

### Target Variable (y)

| Class | Label Index | Real-World Disease |
|---|---|---|
| `healthy` | 0 | No disease — normal crop surface |
| `rust` | 1 | Fungal rust — Puccinia striiformis, P. triticina |
| `blight` | 2 | Early / late blight — Alternaria or Phytophthora |
| `mold` | 3 | Mould or powdery mildew — Erysiphe graminis |

### Data Collection Methods (Real-World Deployment)

| Method | Volume | Cost | Quality |
|---|---|---|---|
| **Farmer mobile app** — crowd-sourced field photos | Very high | Very low | Variable — needs filtering |
| **Drone field surveys** — systematic aerial imagery | High | Medium | High resolution, consistent |
| **Research plot photography** — controlled conditions | Low | High | Excellent — gold standard labels |
| **Public datasets** (PlantVillage, iNaturalist, CGIAR) | Very high | Free | Pre-labelled, peer-reviewed |
| **Agronomist field annotation** | Variable | High per image | Highest accuracy |

### Data Quality Risks

| Risk | Likelihood | Severity | Mitigation Strategy |
|---|---|---|---|
| **Label noise** — farmer misidentifies disease | High | High | Expert validation; 3-agronomist consensus rule |
| **Lighting variation** — field photography at different times of day | High | Medium | Augmentation: brightness and contrast jitter during training |
| **Camera quality** — low-resolution mobile phones | Medium | Medium | Train on multiple resolutions; add image quality pre-check in app |
| **Crop variety bias** — model trained on wheat, deployed on rice | High | High | Collect images across multiple crop species; add crop type as input |
| **Geographic bias** — disease strains differ by region | Medium | High | Include diverse regional datasets; region-specific fine-tuning option |
| **Class imbalance** | Low (balanced here: 120 per class) | Medium | Stratified sampling; class-weighted loss if imbalance appears |
| **Seasonal bias** — disease appearance changes with growth stage | Medium | Medium | Label images with season and growth stage metadata |

### Preprocessing Pipeline

| Step | Operation | Justification |
|---|---|---|
| 1 | Resize to 64×64 px | All images must be the same shape for batch processing |
| 2 | Convert to float32 | Neural networks require floating-point numerical arrays |
| 3 | Normalise ÷ 255 | Scales pixels from [0–255] to [0–1] — stabilises gradient descent |
| 4 | Map labels | Apply CLASS_MAP: `normal → healthy`, `scratch → rust`, etc. |
| 5 | One-hot encode | `blight → [0,0,1,0]` — required for categorical cross-entropy loss |
| 6 | Stratified split | 70% train / 15% val / 15% test — equal class representation in each split |
| 7 | Data augmentation (train only) | Rotation ±20°, horizontal flip, zoom ±10%, brightness jitter ±15% |

---

## Task 5 — Model Recommendation <a name="task-5"></a>

### Recommended Architecture: Custom CNN + Transfer Learning Option

The reference catalog entry for Agriculture explicitly states: `CNN or transfer learning`.

### Why CNN over Other Model Types?

| Model | For this problem? | Reason |
|---|---|---|
| **Custom CNN** | Yes — PRIMARY | Designed for spatial image data; learns texture/colour/shape filters automatically |
| **MobileNetV2 Transfer Learning** | Yes — PRODUCTION | Pre-trained on ImageNet; achieves ~93%+ with only 480 training images |
| Feed-forward (Dense) NN | No | Treats each pixel independently — no spatial awareness; 12,288 inputs × 256 neurons = 3.1M parameters in first layer alone |
| RNN / LSTM | No | Designed for sequential time-series data, not 2D spatial images |
| Autoencoder | No | Unsupervised; we have labels — supervised classification is more appropriate |
| Vision Transformer (ViT) | No | Requires very large datasets (100k+); harder to interpret for field agronomists |

### Custom CNN Architecture (Primary)

```
Input (64 × 64 × 3 RGB image, normalised)
    |
    v
[Block 1]  Conv2D(32 filters, 3×3)  ->  BatchNorm  ->  ReLU  ->  MaxPool(2×2)  ->  Dropout(0.25)
           Output: 32×32×32                                      Output: 32×32×32
    |
    v
[Block 2]  Conv2D(64 filters, 3×3)  ->  BatchNorm  ->  ReLU  ->  MaxPool(2×2)  ->  Dropout(0.25)
           Output: 32×32×64                                      Output: 16×16×64
    |
    v
[Block 3]  Conv2D(128 filters, 3×3) ->  BatchNorm  ->  ReLU  ->  MaxPool(2×2)  ->  Dropout(0.25)
           Output: 16×16×128                                     Output: 8×8×128
    |
    v
[Block 4]  Conv2D(256 filters, 3×3) ->  BatchNorm  ->  ReLU  ->  MaxPool(2×2)  ->  Dropout(0.25)
           Output: 8×8×256                                       Output: 4×4×256
    |
    v
GlobalAveragePooling2D   [4×4×256 -> 256-dimensional vector]
    |
    v
Dense(256)  ->  BatchNorm  ->  ReLU  ->  Dropout(0.50)
    |
    v
Dense(4)  ->  Softmax  [outputs: P(healthy), P(rust), P(blight), P(mold)]
```

### Layer-by-Layer Explanation

| Layer | What It Does | Why It Matters |
|---|---|---|
| **Conv2D** | Applies a small 3×3 filter across the image to detect local visual patterns | Detects edges, colour boundaries, and texture streaks specific to each disease |
| **BatchNormalization** | Normalises activations within each mini-batch | Speeds up training and prevents internal covariate shift |
| **ReLU** | Sets all negative activation values to zero: `f(x) = max(0, x)` | Adds non-linearity — the network can learn curved, complex decision boundaries |
| **MaxPooling2D** | Keeps the maximum value in each 2×2 region | Reduces spatial dimensions, adds position tolerance, reduces computation |
| **Dropout** | Randomly turns off neurons during training | Prevents overfitting — forces the network to learn redundant, robust representations |
| **GlobalAveragePooling2D** | Averages the spatial dimensions to a single vector | More parameter-efficient than Flatten; reduces 4×4×256=4096 to just 256 values |
| **Dense** | Fully-connected layer combining all features | Learns high-level combinations of spatial features for the final decision |
| **Softmax** | Converts 4 raw scores to probabilities summing to 1.0 | Provides interpretable confidence scores per class |

### Training Configuration

| Parameter | Value | Justification |
|---|---|---|
| Optimiser | Adam, lr=0.001 | Adaptive learning rate; works well on small datasets |
| Loss function | Categorical cross-entropy | Standard for multi-class one-hot-encoded targets |
| Batch size | 32 | Good balance of speed and gradient stability |
| Max epochs | 50 | With EarlyStopping to prevent over-training |
| EarlyStopping patience | 10 epochs | Stops if val_loss stalls; restores best weights |
| ReduceLROnPlateau | factor=0.5, patience=5 | Halves lr when training stalls — helps find better minimum |

### Transfer Learning Option (MobileNetV2)

For production deployment, **MobileNetV2 pre-trained on ImageNet** is strongly recommended:

```python
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False        # freeze ImageNet weights initially
# Add custom classification head:
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(4, activation='softmax')(x)
```

**Advantages of Transfer Learning:**
- Requires far fewer training images (even 50 per class can work)
- Faster convergence (typically 10–15 epochs vs 50)
- Consistently achieves 93%+ accuracy on agricultural disease datasets
- Lightweight enough to run on Android / iOS at 30+ FPS

---

## Task 6 — Evaluation Plan <a name="task-6"></a>

### Technical Metrics

| Metric | Formula | Importance for Crop Disease |
|---|---|---|
| **Test Accuracy** | Correct predictions / Total | Overall correct identification rate |
| **Macro Precision** | Mean(TP / (TP + FP)) across 4 classes | Controls false disease alarms → prevents unnecessary pesticide use |
| **Macro Recall** | Mean(TP / (TP + FN)) across 4 classes | Controls missed diseases → catches dangerous spread before it worsens |
| **Macro F1-Score** | Harmonic mean of Precision and Recall | Balanced measure; used when no class should be deprioritised |
| **Confusion Matrix** | Per-class true vs. predicted counts | Shows which diseases are confused with each other |

**Target Values:**
- Test Accuracy ≥ 85%
- Macro F1-Score ≥ 0.80
- Recall for `blight` ≥ 0.88 (most economically damaging — cannot miss it)

### Business Metrics (from `business_kpi_sample.csv`)

| KPI | Current Baseline | AI Target | Measurement Method |
|---|---|---|---|
| Diagnosis response time | 1 to 5 days | Under 5 seconds | App telemetry log |
| Manual processing hours | ~450 hours/month | Under 100 hours/month | Monthly system report |
| Error rate | 6.7% average | Under 3.0% | % AI predictions contradicted by field confirmation |
| Stakeholder satisfaction | 7.1 / 10 | 8.5 / 10 or higher | Quarterly farmer survey |
| Field validation rate | Not tracked | 80% or higher | % AI predictions confirmed by agronomist |
| Annual yield loss prevented | Baseline (unquantified) | Rs 4.2 lakh / 100 acres / season | Crop yield comparison with control group |

### Decision Threshold Tuning

The model outputs a **probability score** for each class. A decision threshold (default 0.50) converts probabilities into a class label.

For agriculture, we recommend **lowering the threshold for disease detection** to prioritise recall over precision:

- Threshold 0.50 → balanced approach
- Threshold 0.40 → higher recall, more false alarms (recommended when disease risk is high-season)
- **Confidence < 0.70 → auto-escalate to agronomist regardless of threshold**

### Business Cost Model

| Prediction Error | Business Cost | Consequence |
|---|---|---|
| False Negative (missed disease) | Rs 5,000 per case | Crop loss, disease spread, emergency remediation |
| False Positive (false alarm) | Rs 300 per case | Unnecessary pesticide application — cost and environmental harm |

At threshold 0.50, total cost per season is minimised (based on cost-curve analysis). Optimal threshold for minimum total business cost is approximately 0.42–0.48 depending on season and crop.

### Possible Failure Cases

| Failure Mode | Trigger | Consequence | Mitigation |
|---|---|---|---|
| Rust classified as Healthy | Low contrast image; poor lighting | Farmer skips treatment; rust spreads to neighbouring plants | Lower disease threshold; show confidence score; agronomist escalation |
| Healthy classified as Blight | Dark shadow on leaf from overcast sky | Unnecessary fungicide application; wasted Rs 300+ | Image quality pre-check; brightness normalisation at inference |
| Unseen crop variety | Model trained on wheat, user photographs rice | All four class probabilities are low and unreliable | Crop-type selector in app; refuse prediction if confidence < 40% |
| Silent model degradation | New disease strain emerges; seasonal shift | Accuracy drops without warning | Monthly holdout set evaluation; automated AUC alert below 0.80 |
| Adversarial input | Photograph of unrelated object | Incorrect high-confidence prediction | Out-of-distribution detection; reject if max probability < 0.55 |

### Human Review and Validation Process

```
Step 1: Model produces class label + confidence score for every image
Step 2: If confidence >= 70% → display result and treatment recommendation to farmer
Step 3: If confidence < 70% → auto-route to agronomist review queue
Step 4: Agronomist confirms or overrides within 4 hours (SLA)
Step 5: All predictions (confirmed and overridden) are logged to the audit trail
Step 6: Override data is collected and used for the next quarterly retraining cycle
Step 7: Monthly performance report reviewed by model owner (agronomist + ML engineer pair)
```

---

## Task 7 — Responsible AI Considerations <a name="task-7"></a>

The reference catalog explicitly flags the key risk as:
> *"Misclassification affecting pesticide usage"*

The following six risks are identified and addressed.

---

### Risk 1: Bias in Training Data (HIGH)

**Description:** The model may be trained predominantly on images of one crop type (e.g. wheat) or one geographic region (e.g. North India). When deployed on different crops (rice, maize, cotton) or different regions (South India, Kenya, Brazil), accuracy degrades — but the model may not communicate this uncertainty to the farmer.

**Why it matters:** Disease strains look visually different across crop species and regions. Puccinia rust on wheat has different visual characteristics than Hemileia rust on coffee. A model trained only on wheat will silently misclassify coffee rust with high confidence.

**Mitigation:**
- Document crop types and geographic regions covered in training data
- Collect and include data from at least 5 different crop species before production deployment
- Display a visible warning when the user photographs a crop type not in the training distribution
- Conduct subgroup accuracy audits split by crop type, region, and camera model

---

### Risk 2: Incorrect Predictions — Misclassification (HIGH)

**Description:** The primary business risk identified in the catalog. A diseased crop classified as healthy results in no treatment, which allows disease to spread. A healthy crop classified as diseased results in unnecessary pesticide application.

**Why it matters:**
- Missed blight can destroy an entire field within 7 days under humid conditions
- Unnecessary fungicide costs Rs 300+ per application and causes soil and water contamination
- Multiple false alarms cause alert fatigue — farmers stop trusting the system

**Mitigation:**
- Tune decision threshold to maximise recall for disease classes (do not miss disease)
- Display confidence score with every prediction
- Require human agronomist review for any prediction with confidence below 70%
- Track precision and recall by disease class monthly
- Set up automated alert if blight recall drops below 88%

---

### Risk 3: Privacy Concerns (MEDIUM)

**Description:** Crop disease data combined with GPS location can reveal sensitive farm economics — a diseased field visible to insurers, landlords, or government agencies may put a farmer at financial or legal risk. If data is stored on a central server, it creates a target for breach.

**Mitigation:**
- Process images on-device (edge AI) wherever possible — do not send raw images to cloud
- Store only anonymised aggregated statistics (disease rate by region — never farm-level data)
- Obtain explicit informed consent before any data sharing with third parties
- Comply with local data protection regulations (India PDPB, GDPR for international deployments)

---

### Risk 4: Over-Reliance on AI (MEDIUM)

**Description:** Farmers who previously exercised their own observational skills may gradually stop trusting their own judgement and follow AI recommendations blindly — even when the photo is blurry, the disease is outside the training distribution, or the confidence score is low.

**Mitigation:**
- App UI always displays: *"AI Confidence: XX%. We recommend confirming with your local agronomist."*
- Educate farmers (via in-app tutorial) on what the model can and cannot do
- Track the override rate (how often agronomists disagree with the model) monthly
- Intentionally make the confidence score prominent — not hidden in fine print

---

### Risk 5: Impact on Small and Marginalised Farmers (MEDIUM)

**Description:** If the model performs better for images taken with expensive high-resolution cameras under ideal lighting, wealthier farmers benefit more — widening the existing digital divide in agriculture. Farmers who are illiterate or lack smartphone access are excluded entirely.

**Mitigation:**
- During model development, test performance specifically on low-resolution (240p) images
- Provide offline inference mode for regions with poor connectivity
- Offer voice-based output (not just text) for illiterate users
- Conduct user testing specifically with marginalised farmer communities before release
- Provide the base model in an SMS-based form for farmers without smartphones

---

### Risk 6: Need for Human Oversight and Model Governance (MEDIUM)

**Description:** Without active monitoring, a model update that degrades performance may not be noticed until a full growing season has passed with widespread misdiagnoses. Concept drift (disease strains evolving, new diseases emerging) silently erodes accuracy over time.

**Mitigation:**
- All model updates require validation on a held-out field-collected test set before deployment
- Monthly automated monitoring dashboard: track accuracy, confidence distribution, and agronomist override rate
- Quarterly model retraining with accumulated field-confirmed override data
- Designate a named model owner (one agronomist + one ML engineer) responsible for every release
- Automated alert to model owner if monthly AUC on monitoring set drops below 0.80

### Responsible AI Risk Summary Table

| Risk | Severity | Likelihood | Priority |
|---|---|---|---|
| Bias by crop type / region | High | High | Immediate — address before production |
| Misclassification → wrong treatment | High | Medium | Immediate — tune threshold + escalation |
| Privacy — GPS and farm data | High | Low | Short-term — on-device inference policy |
| Over-reliance by farmers | Medium | High | Short-term — UI design + education |
| Digital divide — low-income farmers | Medium | Medium | Short-term — SMS and offline modes |
| Silent model degradation over time | High | Medium | Ongoing — monthly monitoring required |

---

## Task 8 — Final Solution Summary <a name="task-8"></a>

---

### Problem

Smallholder farmers globally lose billions of dollars annually to undetected and late-detected crop diseases. The current manual process — waiting 1 to 5 days for an agronomist field visit — is slow, expensive, inconsistent, and geographically inaccessible to most rural farmers. The KPI baseline data shows average manual processing of ~450 hours/month with a 6.7% error rate under the current system.

---

### Proposed AI Solution

A mobile-first CNN image classifier that identifies crop disease type from a single leaf photograph in under 5 seconds. The system integrates into a farmer-facing mobile application and delivers:

1. **Predicted disease class** — `healthy`, `rust`, `blight`, or `mold`
2. **Confidence score** — e.g. "Rust — 91% confidence"
3. **Linked treatment recommendation** — correct fungicide, application rate, and timing
4. **Automatic escalation** — if confidence < 70%, the case is routed to an agronomist review queue

```
Farmer photographs affected crop leaf
              |
              v
Image quality pre-check (blur detection, minimum resolution)
              |
              v
Preprocessing: Resize -> Normalise -> Tensor
              |
              v
CNN Inference: [P(healthy), P(rust), P(blight), P(mold)]
              |
       +------+------+
       |             |
  Confidence >= 70%  Confidence < 70%
       |             |
       v             v
  Display result  Route to agronomist
  + treatment tip   review queue
       |             |
       v             v
  Log to audit   Agronomist confirms
  trail           or overrides (4hr SLA)
       |             |
       +------+------+
              |
              v
    Override data fed back to retraining pipeline
```

---

### Required Data

| Data Type | Source | Minimum Volume |
|---|---|---|
| Crop leaf images (4 classes) | Farmer app + drone + PlantVillage | 500+ images per class per crop type |
| Class labels | Agronomist annotation | One confirmed label per image |
| Crop type metadata | App user input | Mandatory field in photo submission form |
| Optional: GPS, date, weather | App log | Enriches geographic and seasonal analysis |

---

### Model Recommendation

| Option | Expected Accuracy | Recommended For |
|---|---|---|
| **Custom 4-Block CNN** | ~88% test accuracy | Learning from scratch; assignment / prototype |
| **MobileNetV2 Transfer Learning** | ~93%+ test accuracy | Production deployment; mobile inference |

**Training:** Adam optimizer (lr=0.001) · Categorical cross-entropy loss · Batch size 32 · EarlyStopping (patience=10) · ReduceLROnPlateau (patience=5) · Max 50 epochs

---

### Expected Business Impact

| KPI | Current Baseline | Post-AI Target | Estimated Improvement |
|---|---|---|---|
| Diagnosis response time | 1–5 days | Under 5 seconds | 99.9% faster |
| Manual processing hours | ~450 hrs/month | Under 100 hrs/month | −78% |
| Error rate | 6.7% | Under 3.0% | −55% |
| Stakeholder satisfaction | 7.1 / 10 | 8.5 / 10 or higher | +20% |
| Estimated yield loss prevented | Unquantified (baseline) | Rs 4.2 lakh per season per 100 acres | Quantified financial saving |
| Agronomist site visits | ~900 per month | ~200 per month | −78% (focus on complex cases only) |

---

### Risks and Mitigation Plan

| Risk | Severity | Mitigation |
|---|---|---|
| Disease misclassification → wrong pesticide | High | Confidence threshold tuning; recall-optimised thresholds; agronomist escalation |
| Crop/region bias in training data | High | Multi-crop data collection; geographic diversity in training set |
| Over-reliance by farmers | Medium | Confidence score UI; in-app education; agronomist escalation for low confidence |
| Lighting and camera quality variation | Medium | Augmentation during training; image quality pre-check at inference |
| GPS and farm privacy | Medium | On-device inference; anonymised aggregate storage; explicit consent |
| Silent model performance degradation | Medium | Monthly AUC monitoring; automated alert below 0.80; quarterly retraining |

---

### Implementation Roadmap

| Phase | Timeline | Key Deliverable |
|---|---|---|
| Data collection and labelling | Month 1–3 | 2,000+ annotated crop images across 4 crop species |
| Model development and validation | Month 3–5 | CNN achieving ≥ 85% accuracy on held-out test set |
| Mobile app integration | Month 5–7 | Android / iOS app with offline inference mode |
| Pilot deployment — 50 farms | Month 7–9 | Field validation report; farmer feedback collected |
| Scale-up and monitoring dashboard | Month 9–12 | Full regional deployment; live performance monitoring |
| First quarterly retraining | Month 12 | Model updated with 6 months of field override data |

---

*This solution has been designed in accordance with Responsible AI principles. Human oversight, geographic and crop-type fairness, farmer data privacy, and explainable confidence scoring are built into every stage of the pipeline. No automated treatment action is ever taken without human confirmation.*

---

### Solution Architecture Diagram

See `solution_architecture.png` — a 5-layer system diagram showing:

- **Layer 1:** Raw data inputs (Farmer App · Drone · Datasets · Sensors)
- **Layer 2:** Preprocessing pipeline (Resize · Normalise · Encode · Augment · Split · Generate)
- **Layer 3:** CNN architecture (4 Conv blocks · GlobalAvgPool · Dense · Softmax) + Transfer Learning option
- **Layer 4:** Output and thresholding (Confidence Engine · Risk Categories · Treatment Recommendation · Feedback Loop)
- **Layer 5:** Human oversight (Farmer UI · Agronomist Dashboard · Audit · Monitoring · Compliance · Retraining)

---

*End of Report*

# Section 11.8: Ethics & Deployment

> **Source:** Prosise, Ch. 11 - responsible use, bias, consent, liveness; industry context  
> **Prerequisites:** [Section 11.7](./section-07-open-set-classification.md) | [Section 11.1](./section-01-detection-vs-recognition.md)  
> **Glossary:** [classification](../../GLOSSARY.md#classification) | [overfitting](../../GLOSSARY.md#overfitting)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Why Ethics Belongs in an Engineering Chapter

Prosise opens Chapter 11 with Delta's facial recognition pilot and notes face technology is **everywhere and controversial**. Technical accuracy is necessary but not sufficient - engineers ship systems that affect civil liberties, employment, and safety.

This section covers what to evaluate **before** deployment, not after headlines.

> **Humorous analogy:** A face recognizer with 99% accuracy and no consent policy is a sports car with no brakes - impressive specs, catastrophic destination.

> **In plain English:** Build the pipeline in Sections 11.1-11.7, then ask who is affected, whether they agreed, and what happens when the model is wrong.

---

## Detection vs Recognition: Privacy Boundary

| Capability | Identifies person? | Typical regulation sensitivity |
|------------|-------------------|-------------------------------|
| **Detection** | No - counts/locates faces | Lower (still sensitive in EU) |
| **Recognition** | Yes - links to identity | High - biometric data in many jurisdictions |

**GDPR** (EU) and **BIPA** (Illinois) treat biometric identifiers as special category data - consent, retention limits, and purpose limitation apply.

Engineering implication: log **detection events** separately from **identity matches**; minimize storage of raw biometrics.

---

## Consent & Purpose Limitation

| Principle | Engineering practice |
|-----------|---------------------|
| **Informed consent** | UI explains what is captured and why |
| **Purpose limitation** | Don't reuse dorm-access embeddings for marketing |
| **Data minimization** | Store embeddings, not raw video, when possible |
| **Retention limits** | Auto-delete gallery entries after term/employment ends |
| **Right to deletion** | API to remove individual from gallery |

```python
# Example: audit log without storing raw face images
def log_recognition_event(user_id, result, score, store_embedding=False):
    record = {
        'timestamp': datetime.utcnow().isoformat(),
        'user_id': user_id,
        'decision': result,
        'score': round(score, 4),
    }
    if store_embedding:
        record['embedding_hash'] = hashlib.sha256(embedding.tobytes()).hexdigest()
    audit_db.insert(record)
```

Hash embeddings for deduplication audits without reversible storage.

---

## Bias & Demographic Fairness

Face models trained on **imbalanced datasets** (e.g., LFW skew, Prosise's George W. Bush imbalance) perform unevenly across:

- Skin tone
- Gender presentation
- Age
- Camera quality

**Disparate error rates** - higher FAR for some groups - create legal and ethical liability.

| Mitigation | Description |
|------------|-------------|
| Stratified evaluation | Report TAR/FAR per demographic slice |
| Diverse enrollment | Ensure gallery represents served population |
| Threshold per cohort | Controversial - use only with legal review |
| Better embedders | Models trained on balanced global data (e.g., VGGFace2) |

Fairness metric example - max FAR gap across groups:

$$
\Delta_{\text{FAR}} = \max_g \text{FAR}_g - \min_g \text{FAR}_g
$$
> **Readable form:** fairness gap = difference between highest and lowest false accept rate across demographic groups - smaller is more equitable

---

## False Accept vs False Reject: Who Bears Cost?

| Application | Prioritize | Rationale |
|-------------|-----------|-----------|
| Phone unlock | Low FAR | Attacker gains device access |
| Photo auto-tag | Low FRR | User fixes wrong tag |
| Airport watchlist | Low FAR | Wrong person detained |
| Attendance kiosk | Balanced EER | Convenience vs fraud |

Document the **cost matrix** with product and legal stakeholders before picking τ.

---

## Liveness & Spoofing

**Presentation attacks** defeat recognition without breaking the model:

| Attack | Defense |
|--------|---------|
| Printed photo | Texture analysis, depth camera |
| Video replay | Challenge-response (blink, turn head) |
| 3D mask | IR sensors, multi-frame consistency |
| Deepfake | Active liveness + deepfake detectors |

```python
# Conceptual: require motion challenge before high-security match
def liveness_check(video_frames, min_blink=1):
    # Integrate dedicated liveness SDK or optical flow heuristic
    blinks_detected = count_blinks(video_frames)
    return blinks_detected >= min_blink
```

Prosise does not deep-dive liveness - production access control should not skip it.

---

## Surveillance vs Opt-In Use Cases

| Use case | Consent model | Engineering notes |
|----------|---------------|-------------------|
| Phone unlock | Explicit opt-in | On-device processing preferred |
| Photo app tagging | Opt-in per album | Cloud optional |
| Public CCTV identification | Often **no** individual consent | Highest scrutiny - many jurisdictions restrict |
| Workplace entry | Employment agreement | Union/legal review |

**On-device inference** (Core ML, TensorFlow Lite) reduces data exfiltration - Apple Face ID pattern.

---

## Regulatory Landscape (High Level)

| Region | Relevant framework | Engineer takeaway |
|--------|-------------------|-----------------|
| EU | GDPR, AI Act (biometric categorization restrictions) | DPIA, lawful basis, transparency |
| US (state) | BIPA (IL), CCPA | Written consent in some states |
| US (federal) | Sector-specific (FERPA, HIPAA) | Context matters |

This is not legal advice - involve counsel for production deployments.

---

## Security of Biometric Data

Unlike passwords, **you cannot rotate a face** if the gallery leaks.

| Control | Implementation |
|---------|----------------|
| Encryption at rest | AES-256 on gallery DB |
| Encryption in transit | TLS 1.2+ for API calls |
| Access control | RBAC on enrollment endpoints |
| No plaintext logs | Strip embeddings from debug logs |
| Key rotation | Rotate API keys, not faces |

```python
from cryptography.fernet import Fernet

key = Fernet.generate_key()
f = Fernet(key)
encrypted_gallery = f.encrypt(json.dumps(gallery).encode())
```

---

## Transparency & Human Override

Users and subjects should know:

1. That face technology is in use
2. How to contest a false match
3. How to request deletion

Engineering: expose **confidence score** to operators, not just binary allow/deny. Human review for borderline scores:

$$
\text{human review if } \tau_{\text{low}} \leq \text{sim} < \tau_{\text{high}}
$$
> **Readable form:** route ambiguous matches (similarity between low and high thresholds) to a human instead of automatic decision

---

## Monitoring in Production

| Signal | Alert if |
|--------|----------|
| Score distribution drift | Mean similarity drops week-over-week |
| Unknown rate spike | Possible attack or lighting change |
| Per-camera FAR | One camera misconfigured |
| Enrollment failures | Detector missing faces |

Track **outcomes** (false accepts reported by users), not just model scores.

---

## Responsible Development Checklist

- [ ] Legal review for jurisdiction and use case
- [ ] Consent flow documented and implemented
- [ ] Stratified evaluation on representative data
- [ ] Threshold chosen with FAR/TAR tradeoff sign-off
- [ ] Liveness for high-security paths
- [ ] Gallery encryption and deletion API
- [ ] Incident response plan for false arrests/denials
- [ ] Public signage where cameras operate

---

## Industry Trends

Meta discontinued automatic face tagging (2021) citing societal concerns. Microsoft limited Azure Face **identification** features pending regulation. These shifts signal that **technical capability ≠ social license to deploy**.

Engineers who understand detection, embeddings, and open-set rejection can also advocate for **not deploying** when risks outweigh benefits.

---

## Self-Check

1. Why is recognition more regulated than detection?
2. Name two mitigations for demographic bias in face systems.
3. What is a presentation attack and one defense?
4. Why can't you "reset" a compromised biometric like a password?
5. When should human review replace automatic threshold decisions?

---

## References

- Prosise, *Applied ML and AI for Engineers*, Ch. 11 - societal context
- [NIST Face Recognition Vendor Test](https://www.nist.gov/programs-projects/face-recognition-vendor-test-frvt)
- [EU GDPR - biometric data](https://gdpr-info.eu/)
- [Buolamwini & Gebru, Gender Shades, 2018](http://gendershades.org/)
- [Microsoft Responsible AI face recognition guidance](https://www.microsoft.com/en-us/ai/responsible-ai)
- [GLOSSARY.md](../../GLOSSARY.md) | [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Previous:** [Section 11.7](./section-07-open-set-classification.md) | **Next:** [Lab 11](./section-lab-11-face-detection-and-recognition-pipeline.md)

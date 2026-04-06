# Use Cases Avances — Cybersecurity IDS

> Ce document decrit les extensions possibles du projet pour se demarquer.
> Priorise par faisabilite dans le cadre du projet (3 semaines, 4 membres).

---

## Use Case 1: OpenRouter — Explication LLM des alertes

**Concept :** Quand le modele ML detecte une intrusion, un LLM genere automatiquement une explication en langage naturel de la menace.

**Pourquoi c'est pertinent :**
- Les analystes securite ont besoin de comprendre *pourquoi* une alerte a ete declenchee
- Ca transforme le projet d'un simple classificateur en un outil d'aide a la decision
- Ca montre qu'on sait integrer des APIs modernes dans un pipeline ML

**Implementation :**

```python
# src/utils/llm_enrichment.py
import requests
import os

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

def explain_prediction(features: dict, prediction: int, confidence: float, shap_top_features: list) -> str:
    """Generate LLM explanation of an IDS alert."""

    prompt = f"""You are a cybersecurity analyst. Analyze this network session flagged by our IDS:

Session Details:
- Protocol: {features['protocol_type']}
- Packet Size: {features['network_packet_size']} bytes
- Login Attempts: {features['login_attempts']} (Failed: {features['failed_logins']})
- Session Duration: {features['session_duration']:.1f}s
- Encryption: {features['encryption_used']}
- IP Reputation Score: {features['ip_reputation_score']:.2f}
- Browser: {features['browser_type']}
- Unusual Time Access: {'Yes' if features['unusual_time_access'] else 'No'}

ML Model Prediction: {'ATTACK DETECTED' if prediction == 1 else 'Normal Traffic'}
Confidence: {confidence:.1%}

Top contributing features (SHAP): {shap_top_features}

Provide:
1. Threat assessment (1-2 sentences)
2. Likely attack type (brute force, DDoS, reconnaissance, etc.)
3. Recommended action (block, monitor, investigate)
4. MITRE ATT&CK technique ID if applicable
"""

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "deepseek/deepseek-r1",  # Cost-effective reasoning model
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 300,
        }
    )
    return response.json()["choices"][0]["message"]["content"]
```

**Cout estime :** ~$0.50-2/jour pour un usage modere (100 alertes/jour)

**Modeles recommandes sur OpenRouter :**
- `deepseek/deepseek-r1` — Bon reasoning, peu cher
- `openai/o3-mini` — Tres bon pour l'analyse technique
- `openrouter/auto` — Selection auto du meilleur rapport qualite/prix

---

## Use Case 2: CrowdSec — Threat Intelligence communautaire

**Concept :** Enrichir les predictions du modele ML avec les donnees de reputation IP de CrowdSec (base communautaire de 70,000+ utilisateurs, 10M signaux/jour).

**Pourquoi c'est pertinent :**
- Validation croisee : comparer les predictions ML vs intelligence communautaire
- Feature engineering : ajouter le score CrowdSec comme feature
- Production-ready : c'est exactement comme ca que les vrais IDS fonctionnent

**Implementation :**

```python
# src/utils/crowdsec_enrichment.py
import requests
import os

CROWDSEC_API_KEY = os.getenv("CROWDSEC_API_KEY")
CROWDSEC_API_URL = "https://cti.api.crowdsec.net/v2"

def get_ip_reputation(ip_address: str) -> dict:
    """Query CrowdSec CTI for IP reputation."""
    response = requests.get(
        f"{CROWDSEC_API_URL}/smoke/{ip_address}",
        headers={"x-api-key": CROWDSEC_API_KEY}
    )

    if response.status_code == 200:
        data = response.json()
        return {
            "ip": ip_address,
            "classification": data.get("classifications", {}).get("classifications", []),
            "behaviors": [b["name"] for b in data.get("behaviors", [])],
            "attack_details": data.get("attack_details", []),
            "reputation": data.get("reputation", "unknown"),
            "confidence": data.get("confidence", 0),
            "last_seen": data.get("history", {}).get("last_seen"),
        }
    return {"ip": ip_address, "reputation": "unknown", "confidence": 0}

# Classifications CrowdSec :
# - malicious : activement malveillant
# - suspicious : signale mais pas confirme
# - known : flagge mais sans conclusion
# - benign : pas de menace
# - safe : service legitime
```

**Scenario de comparaison pour le rapport :**

```python
def compare_ml_vs_crowdsec(predictions_df):
    """
    Compare ML model predictions against CrowdSec ground truth.
    Excellent for the report's evaluation section.
    """
    results = {
        "ml_attack_crowdsec_malicious": 0,  # Accord total
        "ml_attack_crowdsec_benign": 0,     # Faux positif ML ?
        "ml_normal_crowdsec_malicious": 0,  # Faux negatif ML !
        "ml_normal_crowdsec_benign": 0,     # Accord total
    }

    for _, row in predictions_df.iterrows():
        ml_attack = row["ml_prediction"] == 1
        cs_malicious = row["crowdsec_reputation"] == "malicious"

        if ml_attack and cs_malicious:
            results["ml_attack_crowdsec_malicious"] += 1
        elif ml_attack and not cs_malicious:
            results["ml_attack_crowdsec_benign"] += 1
        elif not ml_attack and cs_malicious:
            results["ml_normal_crowdsec_malicious"] += 1
        else:
            results["ml_normal_crowdsec_benign"] += 1

    return results
```

**Note :** Free tier CrowdSec = ~50 requetes/jour. Suffisant pour le projet.

---

## Use Case 3: MITRE ATT&CK Mapping

**Concept :** Mapper les detections du modele sur le framework MITRE ATT&CK, le standard de l'industrie pour classifier les attaques.

**Pourquoi c'est pertinent :**
- Ca parle directement aux professionnels de la securite
- Ca donne un contexte riche aux alertes (tactique, technique, procedure)
- Ca montre une comprehension du domaine au-dela du ML

**Mapping pour notre dataset :**

```python
ATTACK_PATTERNS = {
    "brute_force": {
        "condition": lambda row: row["failed_logins"] > 5 and row["login_attempts"] > 8,
        "mitre_id": "T1110",
        "mitre_name": "Brute Force",
        "tactic": "Credential Access",
    },
    "reconnaissance": {
        "condition": lambda row: row["session_duration"] < 10 and row["network_packet_size"] < 100,
        "mitre_id": "T1046",
        "mitre_name": "Network Service Discovery",
        "tactic": "Discovery",
    },
    "suspicious_access": {
        "condition": lambda row: row["unusual_time_access"] == 1 and row["ip_reputation_score"] > 0.7,
        "mitre_id": "T1078",
        "mitre_name": "Valid Accounts",
        "tactic": "Initial Access",
    },
    "unencrypted_exfiltration": {
        "condition": lambda row: row["encryption_used"] == "None" and row["network_packet_size"] > 1200,
        "mitre_id": "T1048",
        "mitre_name": "Exfiltration Over Alternative Protocol",
        "tactic": "Exfiltration",
    },
}
```

---

## Use Case 4: Adversarial Robustness Testing (Bonus)

**Concept :** Tester si un attaquant peut craft du trafic malveillant qui trompe le modele (evasion attacks).

```python
# Utiliser Adversarial Robustness Toolbox (ART)
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import SklearnClassifier

# Wrapper pour le modele
classifier = SklearnClassifier(model=trained_model, clip_values=(0, 1))

# Attaque FGSM (Fast Gradient Sign Method)
attack = FastGradientMethod(estimator=classifier, eps=0.2)
X_adversarial = attack.generate(X_test)

# Comparer les performances
original_acc = model.score(X_test, y_test)
adversarial_acc = model.score(X_adversarial, y_test)
robustness_gap = original_acc - adversarial_acc
```

---

## Priorisation recommandee

| Use Case | Effort | Impact sur la note | Priorite |
|---|---|---|---|
| SHAP/LIME (deja dans le scope) | Faible | Tres haut (10%) | OBLIGATOIRE |
| MITRE ATT&CK mapping | Faible | Haut (bonus) | Recommande |
| OpenRouter explication LLM | Moyen | Haut (bonus + demo wow factor) | Recommande |
| CrowdSec validation croisee | Moyen | Haut (bonus report section) | Si le temps permet |
| Adversarial testing | Moyen-Haut | Moyen (bonus) | Si le temps permet |

---

## Architecture globale avec use cases

```
                    Cybersecurity Intrusion Detection System
                    ========================================

  [Dataset CSV] ──► [Preprocessing] ──► [Feature Engineering]
                                              │
                          ┌───────────────────┼───────────────────┐
                          │                   │                   │
                    [ML Baselines]      [DNN Keras]        [Ensemble]
                    LR / RF / XGB       128→64→32→1        Voting/Stack
                          │                   │                   │
                          └───────────────────┼───────────────────┘
                                              │
                                    [Evaluation & Comparison]
                                    Accuracy/F1/AUC/ROC
                                              │
                          ┌───────────────────┼───────────────────┐
                          │                   │                   │
                    [SHAP/LIME]        [MITRE ATT&CK]      [CrowdSec]
                    Interpretability    Attack Mapping     IP Validation
                          │                   │                   │
                          └───────────────────┼───────────────────┘
                                              │
                                    [OpenRouter LLM]
                                    Alert Explanation
                                              │
                                    [FastAPI + Docker]
                                    REST API Deployment
                                              │
                                      [MLflow Tracking]
                                    Experiment Logging
```

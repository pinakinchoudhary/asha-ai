"""
Synthetic data generators for ASHA Copilot.

Generates realistic Indian maternal/child health records calibrated to NFHS-5 distributions.
All generators return PySpark DataFrames for direct Delta Lake writes.
"""

import random
import uuid
from datetime import date, timedelta

# ---------------------------------------------------------------------------
# Static reference data (avoids external downloads at generation time)
# ---------------------------------------------------------------------------

STATES_DISTRICTS_VILLAGES = {
    "Uttar Pradesh": {
        "Varanasi": ["Rampur", "Sultanpur", "Bhadohi", "Chandauli", "Ramnagar",
                      "Sarnath", "Cholapur", "Kashi Vidyapeeth"],
        "Lucknow": ["Bakshi Ka Talab", "Chinhat", "Mohanlalganj", "Sarojini Nagar"],
        "Gorakhpur": ["Campierganj", "Sahjanwa", "Chauri Chaura", "Bansgaon"],
    },
    "Madhya Pradesh": {
        "Sehore": ["Sehore", "Ashta", "Ichhawar", "Budni"],
        "Bhopal": ["Berasia", "Phanda", "Kolar", "Ratibad"],
    },
    "Rajasthan": {
        "Tonk": ["Tonk", "Deoli", "Uniara", "Malpura", "Peeplu", "Dooni"],
        "Jaipur": ["Sanganer", "Phagi", "Kotputli", "Amber"],
    },
    "Bihar": {
        "Patna": ["Phulwari Sharif", "Danapur", "Khagaul", "Maner", "Bihta"],
        "Jehanabad": ["Jehanabad", "Kako", "Ghosi", "Makhdumpur"],
    },
}

# Village → nearest PHC mapping
VILLAGE_PHC_MAP = {
    "Rampur": "PHC-001", "Sultanpur": "PHC-001", "Bhadohi": "PHC-001",
    "Chandauli": "PHC-001", "Ramnagar": "CHC-001", "Sarnath": "CHC-001",
    "Cholapur": "CHC-001", "Kashi Vidyapeeth": "CHC-001",
    "Sehore": "PHC-002", "Ashta": "PHC-002", "Ichhawar": "PHC-002",
    "Budni": "PHC-002",
    "Tonk": "PHC-003", "Deoli": "CHC-002", "Uniara": "PHC-003",
    "Malpura": "PHC-003", "Peeplu": "CHC-002", "Dooni": "CHC-002",
    "Phulwari Sharif": "PHC-004", "Danapur": "PHC-004", "Khagaul": "PHC-004",
    "Maner": "PHC-004",
    "Jehanabad": "PHC-005", "Kako": "PHC-005", "Ghosi": "PHC-005",
    "Makhdumpur": "PHC-005",
}

FIRST_NAMES = [
    "Sunita", "Lakshmi", "Meena", "Kavita", "Rani", "Pooja", "Geeta",
    "Savitri", "Parvati", "Radha", "Sita", "Anita", "Rekha", "Usha",
    "Kamla", "Babita", "Neelam", "Saroj", "Pushpa", "Maya", "Renu",
    "Sarita", "Manju", "Kiran", "Priya", "Nirmala", "Shanti", "Guddi",
    "Phoolmati", "Champa", "Munni", "Dulari", "Pinky", "Ruby", "Sapna",
    "Asha", "Devi", "Kumari", "Basanti", "Janki",
]

SURNAMES_BY_STATE = {
    "Uttar Pradesh": ["Yadav", "Singh", "Verma", "Gupta", "Sharma", "Patel",
                       "Mishra", "Tripathi", "Pandey", "Chauhan"],
    "Madhya Pradesh": ["Gond", "Malviya", "Patel", "Thakur", "Ahirwar",
                        "Rajput", "Sahu", "Lodhi"],
    "Rajasthan": ["Meena", "Gurjar", "Jat", "Sharma", "Choudhary",
                   "Rajput", "Bairwa", "Prajapat"],
    "Bihar": ["Kumar", "Devi", "Yadav", "Paswan", "Mandal", "Thakur",
              "Singh", "Prasad", "Sah"],
}

CASTE_CATEGORIES = ["General", "SC", "ST", "OBC"]
CASTE_WEIGHTS = [0.15, 0.20, 0.10, 0.55]  # approximate national distribution

RISK_FLAGS = [
    "Previous LSCS", "Chronic Hypertension", "Gestational Diabetes",
    "Severe Anemia History", "Previous Stillbirth", "Rh Negative",
    "Multiple Pregnancy", "Teenage Pregnancy", "Grand Multipara",
]

BLOOD_GROUPS = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]

# Hinglish clinical notes templates
CLINICAL_NOTES_NORMAL = [
    "Patient theek hai, koi shikayat nahi. BP normal, baby ki movement theek hai.",
    "ANC visit routine. Weight badh raha hai normally. Iron tablet le rahi hai.",
    "Patient ne bataya sab theek hai. Khana theek se kha rahi hai. No complaints.",
    "Routine checkup. BP normal range mein. Baby ki heartbeat sunai di.",
    "Visit mein koi problem nahi mili. Patient active hai, diet follow kar rahi hai.",
    "Patient swasth hai. IFA tablets regular le rahi hai. Next visit schedule kiya.",
]

CLINICAL_NOTES_MODERATE = [
    "Patient ko halka chakkar aa raha hai. Hemoglobin kam hai, iron tablet dose badhaya.",
    "Thodi kamzori hai. Khana theek se nahi kha pa rahi. Weight nahi badh raha.",
    "BP thoda upar hai. Patient ko aaraam karne ko bola. Salt kam karne ko bola.",
    "Patient ko kabhi kabhi sir dard hota hai. BP monitor karna hai.",
    "Hemoglobin kam hai. Anemia ki shikayat. Diet counseling di gayi.",
    "Patient ne bataya pair mein halki sujan hai. BP check kiya, thoda upar tha.",
]

CLINICAL_NOTES_DANGER = [
    "Patient ko kal raat se severe headache aur chakkar aa raha hai. BP check kiya toh high tha. Pairo mein sujan hai.",
    "Patient ko vaginal bleeding ho rahi hai. Turant PHC bheja ja raha hai.",
    "Patient ko tez bukhar hai aur kaanp rahi hai. Bahut kamzori. Emergency refer.",
    "Bachche ki movement bahut kam ho gayi hai. Patient pareshan hai. PHC refer.",
    "Patient ko daure aa rahe hain. Behosh ho gayi thi. Emergency ambulance bulaya.",
    "Bahut zyada bleeding. Patient weak hai. BP gir raha hai. Emergency.",
]

# UIP Immunization Schedule (vaccine_code, due_age_weeks, catch_up_max_weeks)
UIP_SCHEDULE = [
    ("BCG", 0, 52),
    ("OPV-0", 0, 2),
    ("Hep-B Birth", 0, 4),
    ("OPV-1", 6, 52),
    ("Pentavalent-1", 6, 52),
    ("Rota-1", 6, 52),
    ("fIPV-1", 6, 52),
    ("PCV-1", 6, 52),
    ("OPV-2", 10, 52),
    ("Pentavalent-2", 10, 52),
    ("Rota-2", 10, 52),
    ("OPV-3", 14, 52),
    ("Pentavalent-3", 14, 52),
    ("Rota-3", 14, 52),
    ("fIPV-2", 14, 52),
    ("PCV-2", 14, 52),
    ("MR-1", 39, 78),  # 9 months
    ("JE-1", 39, 78),
    ("PCV-Booster", 39, 78),
    ("DPT-B1", 70, 104),  # 16 months
    ("OPV-Booster", 70, 104),
    ("MR-2", 70, 104),
    ("JE-2", 70, 104),
]


def _random_location():
    """Pick a random state/district/village."""
    state = random.choice(list(STATES_DISTRICTS_VILLAGES.keys()))
    district = random.choice(list(STATES_DISTRICTS_VILLAGES[state].keys()))
    village = random.choice(STATES_DISTRICTS_VILLAGES[state][district])
    phc_id = VILLAGE_PHC_MAP.get(village, "PHC-001")
    return state, district, village, phc_id


def generate_patients(spark, n: int = 500):
    """Generate synthetic patient registry DataFrame."""
    rows = []
    today = date.today()
    for _ in range(n):
        state, district, village, phc_id = _random_location()
        age = random.randint(18, 40)
        surname = random.choice(SURNAMES_BY_STATE.get(state, ["Devi"]))
        name = f"{random.choice(FIRST_NAMES)} {surname}"
        caste = random.choices(CASTE_CATEGORIES, weights=CASTE_WEIGHTS, k=1)[0]
        bpl = random.random() < 0.30
        gravida = random.randint(1, 5)
        para = max(0, gravida - random.randint(0, 2))
        abortions = gravida - para - 1 if gravida > para else 0
        # LMP: 4-36 weeks ago
        weeks_ago = random.randint(4, 36)
        lmp = today - timedelta(weeks=weeks_ago)
        edd = lmp + timedelta(days=280)
        # Risk flags — 20% of patients have at least one
        flags = []
        if random.random() < 0.20:
            flags = random.sample(RISK_FLAGS, k=random.randint(1, 2))

        rows.append({
            "patient_id": str(uuid.uuid4()),
            "name": name,
            "age": age,
            "husband_name": f"{random.choice(FIRST_NAMES[:10])}ji {surname}",
            "state": state,
            "district": district,
            "village": village,
            "nearest_phc_id": phc_id,
            "caste_category": caste,
            "bpl_status": bpl,
            "blood_group": random.choice(BLOOD_GROUPS),
            "gravida": gravida,
            "para": para,
            "abortions": abortions,
            "lmp": str(lmp),
            "edd": str(edd),
            "risk_flags": flags,
            "aadhaar_registered": random.random() < 0.75,
            "bank_account": random.random() < 0.65,
            "phone": f"+91-{random.randint(7000000000, 9999999999)}",
            "registered_date": str(today - timedelta(days=random.randint(0, weeks_ago * 7))),
        })
    return spark.createDataFrame(rows)


def generate_visits(spark, patients_df, avg_visits: int = 3):
    """Generate synthetic ANC visit records."""
    rows = []
    today = date.today()
    patients = patients_df.collect()

    for p in patients:
        lmp = date.fromisoformat(p["lmp"])
        edd = date.fromisoformat(p["edd"])
        n_visits = random.randint(1, min(4, avg_visits + 1))
        gestational_span = (min(edd, today) - lmp).days

        if gestational_span <= 0:
            continue

        for v in range(1, n_visits + 1):
            visit_day = int(gestational_span * v / (n_visits + 1))
            visit_date = lmp + timedelta(days=visit_day)
            if visit_date > today:
                break
            gest_weeks = visit_day // 7

            # Vitals calibrated to NFHS-5
            roll = random.random()
            if roll < 0.07:  # 5-8% danger
                bp_s = random.randint(145, 180)
                bp_d = random.randint(95, 120)
                hb = round(random.uniform(5.0, 7.5), 1)
                notes = random.choice(CLINICAL_NOTES_DANGER)
                danger = random.sample(
                    ["severe headache", "vaginal bleeding", "convulsions",
                     "decreased fetal movement", "high fever", "severe abdominal pain"],
                    k=random.randint(1, 2),
                )
            elif roll < 0.25:  # ~18% moderate issues
                bp_s = random.randint(130, 145)
                bp_d = random.randint(82, 95)
                hb = round(random.uniform(7.0, 9.0), 1)
                notes = random.choice(CLINICAL_NOTES_MODERATE)
                danger = []
            else:  # ~75% normal
                bp_s = random.randint(100, 130)
                bp_d = random.randint(60, 82)
                # 55% mild-moderate anemia per NFHS-5
                if random.random() < 0.55:
                    hb = round(random.uniform(9.0, 10.9), 1)
                else:
                    hb = round(random.uniform(11.0, 14.0), 1)
                notes = random.choice(CLINICAL_NOTES_NORMAL)
                danger = []

            weight = round(random.uniform(40.0, 75.0) + gest_weeks * 0.3, 1)
            fundal = round(gest_weeks * 1.0 + random.uniform(-2, 2), 1) if gest_weeks > 12 else None
            temp = round(random.uniform(36.2, 37.2), 1)
            if "bukhar" in notes or "fever" in notes.lower():
                temp = round(random.uniform(38.5, 40.0), 1)

            rows.append({
                "visit_id": str(uuid.uuid4()),
                "patient_id": p["patient_id"],
                "visit_number": v,
                "visit_date": str(visit_date),
                "gestational_age_weeks": gest_weeks,
                "weight_kg": weight,
                "fundal_height_cm": fundal,
                "bp_systolic": bp_s,
                "bp_diastolic": bp_d,
                "hemoglobin_g_dl": hb,
                "temperature_c": temp,
                "danger_signs_observed": danger,
                "clinical_notes": notes,
                "referral_flag": len(danger) > 0 or bp_s >= 140 or hb < 7.0,
            })
    return spark.createDataFrame(rows)


def generate_immunizations(spark, patients_df):
    """Generate child immunization records (for children already born)."""
    rows = []
    today = date.today()
    patients = patients_df.collect()

    for p in patients:
        edd = date.fromisoformat(p["edd"])
        # Only generate immunizations for babies already born (EDD in past)
        if edd >= today:
            continue
        # Simulate some births slightly before EDD
        birth_date = edd - timedelta(days=random.randint(-14, 7))
        if birth_date >= today:
            continue

        child_id = str(uuid.uuid4())
        child_age_weeks = (today - birth_date).days // 7

        for vaccine_code, due_weeks, catchup_weeks in UIP_SCHEDULE:
            if due_weeks > child_age_weeks + 4:
                break  # Vaccine not yet due
            due_date = birth_date + timedelta(weeks=due_weeks)
            # 80% administered on time, 10% late, 10% missed
            roll = random.random()
            if roll < 0.80:
                admin_date = due_date + timedelta(days=random.randint(0, 7))
                if admin_date > today:
                    admin_date = None
            elif roll < 0.90:
                admin_date = due_date + timedelta(days=random.randint(14, 60))
                if admin_date > today:
                    admin_date = None
            else:
                admin_date = None  # missed

            missed = admin_date is None and today > due_date + timedelta(weeks=2)

            rows.append({
                "record_id": str(uuid.uuid4()),
                "child_id": child_id,
                "mother_id": p["patient_id"],
                "child_name": f"Baby of {p['name']}",
                "date_of_birth": str(birth_date),
                "vaccine_code": vaccine_code,
                "due_date": str(due_date),
                "administered_date": str(admin_date) if admin_date else None,
                "missed_flag": missed,
                "village": p["village"],
                "nearest_phc_id": p["nearest_phc_id"],
            })

    return spark.createDataFrame(rows) if rows else None


def generate_scheme_applications(spark, patients_df):
    """Generate partial welfare scheme application records."""
    rows = []
    patients = patients_df.collect()

    for p in patients:
        # PMMVY — eligible if first pregnancy + age >= 19 + aadhaar + bank
        if p["para"] == 0 and p["age"] >= 19:
            missing = []
            if not p["aadhaar_registered"]:
                missing.append("Aadhaar card")
            if not p["bank_account"]:
                missing.append("Bank account")

            status = "Eligible" if not missing else "Pending Documents"
            rows.append({
                "application_id": str(uuid.uuid4()),
                "patient_id": p["patient_id"],
                "scheme_id": "PMMVY",
                "eligibility_status": status,
                "missing_documents": missing,
                "disbursement_amount": 5000,
                "applied_date": p["registered_date"],
            })

        # JSY — BPL or SC/ST
        if p["bpl_status"] or p["caste_category"] in ("SC", "ST"):
            is_lps = p["state"] in ("Uttar Pradesh", "Bihar", "Madhya Pradesh", "Rajasthan")
            amount = 1400 if is_lps else 700
            rows.append({
                "application_id": str(uuid.uuid4()),
                "patient_id": p["patient_id"],
                "scheme_id": "JSY",
                "eligibility_status": "Eligible",
                "missing_documents": [],
                "disbursement_amount": amount,
                "applied_date": p["registered_date"],
            })

        # JSSK — universal for all pregnant women
        rows.append({
            "application_id": str(uuid.uuid4()),
            "patient_id": p["patient_id"],
            "scheme_id": "JSSK",
            "eligibility_status": "Eligible",
            "missing_documents": [],
            "disbursement_amount": 0,  # cashless services, not cash
            "applied_date": p["registered_date"],
        })

    return spark.createDataFrame(rows)

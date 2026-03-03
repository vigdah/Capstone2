# Capstone2 Project

## Project History

### 2026-03-03
- Created local git repository in `C:\Users\v3ct0\OneDrive\Desktop\Capstone2`
- Added `.gitignore` to exclude `.idea/` (IDE config folder)
- Made initial commit
- Pushed repository to GitHub: https://github.com/vigdah/Capstone2
- Implemented full project scaffold: Android shell, ML pipeline, Spring Boot backend

---

## Project Overview

**Goal:** AI-driven Android app to detect data-exfiltration malware through behavioral analysis.

The app collects per-app network and usage statistics in the background, feeds them into an on-device ML model (XGBoost+MLP ensemble, exported to ONNX), and alerts the user when an app exhibits malicious behavior patterns.

**GitHub:** https://github.com/vigdah/Capstone2

---

## Tech Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| Android app | Kotlin + Jetpack Compose | Min SDK 30 (Android 11) |
| UI toolkit | Material 3 | latest |
| Navigation | Navigation Compose | latest |
| Local DB | Room + SQLCipher | latest |
| Background work | Foreground Service + WorkManager | latest |
| On-device ML | ONNX Runtime Mobile | latest |
| HTTP client | Retrofit + OkHttp | latest |
| DI | Hilt | latest |
| Backend | Spring Boot (Kotlin) | 3.x |
| Cloud DB | Firebase Firestore | latest |
| ML training | Python: XGBoost, PyTorch MLP | latest |
| ML export | ONNX / skl2onnx / onnxmltools | latest |

---

## Build Phases

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 0 | DONE | Update CLAUDE.md + memory |
| Phase 1 | DONE (scaffold) | Android shell: UI + data collection + local storage |
| Phase 2 | DONE (scaffold) | ML pipeline: train XGBoost+MLP, export to ONNX |
| Phase 3 | Pending | Wire ONNX model into Android (ModelRunner.kt) |
| Phase 4 | DONE (scaffold) | Spring Boot backend + Firebase Firestore |
| Phase 5 | Pending | Integration, alerts, polish |

---

## Project Structure

```
Capstone2/
├── android/                    # Android app (Kotlin + Compose)
│   └── app/src/main/
│       ├── java/com/capstone2/malwaredetector/
│       │   ├── ui/screens/     # HomeScreen, AppListScreen, AlertsScreen, AppDetailScreen, SettingsScreen
│       │   ├── ui/components/  # Reusable Compose components
│       │   ├── ui/theme/       # Material 3 theme
│       │   ├── ui/navigation/  # NavGraph.kt
│       │   ├── data/local/     # AppDatabase.kt, entities/, dao/
│       │   ├── data/repository/# BehaviorRepository, AlertRepository
│       │   ├── collection/     # DataCollectionService, NetworkStatsCollector, etc.
│       │   ├── features/       # FeatureExtractor.kt
│       │   ├── ml/             # ModelRunner.kt (ONNX Runtime wrapper)
│       │   ├── alerts/         # AlertManager.kt
│       │   └── di/             # Hilt modules
│       ├── res/                # Android resources
│       └── AndroidManifest.xml
├── backend/                    # Spring Boot middleware (Kotlin)
│   └── src/main/kotlin/com/capstone2/backend/
│       ├── controller/         # DetectionController, AlertController, ModelController
│       ├── service/            # DetectionService, AlertService
│       ├── repository/         # FirestoreRepository
│       └── config/             # FirebaseConfig
├── ml/                         # ML training pipeline (Python)
│   ├── data/raw/               # Downloaded datasets
│   ├── notebooks/              # Exploratory analysis
│   └── src/
│       ├── ingest.py, features.py, train_xgboost.py
│       ├── train_mlp.py, ensemble.py, evaluate.py, export_onnx.py
│       └── requirements.txt
└── CLAUDE.md
```

---

## Key Architectural Decisions

1. **On-device inference with ONNX Runtime Mobile** — keeps inference private, works offline. Model loaded from `android/app/src/main/assets/model.onnx`.
2. **Foreground Service for data collection** — required on Android 11+ for persistent background operation. DataCollectionService collects every N seconds (configurable).
3. **SQLCipher-encrypted Room DB** — all local behavior/detection data encrypted at rest using key from Android Keystore.
4. **Stacked ensemble: XGBoost + MLP** — XGBoost handles tabular features well; MLP provides complementary non-linear representations. Meta-learner combines both.
5. **Spring Boot backend** — middleware for cloud sync, future model update distribution, and server-side alerting. Firebase Firestore for document storage.
6. **Permission strategy** — PACKAGE_USAGE_STATS requires user to manually grant in Settings; onboarding screen explains why.

---

## Key File Paths

| File | Purpose |
|------|---------|
| `android/app/src/main/java/.../collection/DataCollectionService.kt` | Foreground service — heart of data collection |
| `android/app/src/main/java/.../features/FeatureExtractor.kt` | Converts raw stats to model input vector |
| `android/app/src/main/java/.../ml/ModelRunner.kt` | ONNX Runtime inference wrapper |
| `android/app/src/main/java/.../data/local/AppDatabase.kt` | Room + SQLCipher encrypted database |
| `android/app/src/main/java/.../ui/navigation/NavGraph.kt` | App navigation structure |
| `android/app/src/main/assets/model.onnx` | Trained model (generated in Phase 2) |
| `ml/src/train_xgboost.py` | XGBoost model training |
| `ml/src/train_mlp.py` | MLP model training |
| `ml/src/export_onnx.py` | Export to ONNX for mobile |
| `backend/src/main/kotlin/.../controller/DetectionController.kt` | Main API endpoint |

---

## ML Feature Vector

Per app per time window (60-second windows), normalized to [0,1]:
- `bytes_sent`, `bytes_received`, `bytes_ratio`
- `packets_sent`, `packets_received`
- `connections_count`, `unique_destinations`
- `upload_rate`, `download_rate`
- `foreground_time_ratio`, `background_time_ratio`
- `battery_consumption_rate`
- Target: 20–50 features total

---

## Notes

- Recommended dataset: **CICAndMal2017** (network traffic-based behavioral features matching our collection approach)
- Model target size: < 10MB after INT8 quantization
- Detection threshold: configurable in SettingsScreen (default: 0.7 confidence)
- Backend API base: `POST /api/detections`, `GET /api/alerts`, `GET /api/model/latest`
